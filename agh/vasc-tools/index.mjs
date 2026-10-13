import { execFile } from 'node:child_process'
import { join } from 'node:path'

const Kind = Symbol.for('TypeBox.Kind')
const Optional = Symbol.for('TypeBox.Optional')
const optional = (schema) => ({ ...schema, [Optional]: 'Optional' })
const string = (options = {}) => ({ [Kind]: 'String', type: 'string', ...options })
const integer = (options = {}) => ({ [Kind]: 'Integer', type: 'integer', ...options })
const literalUnion = (...values) => ({
  [Kind]: 'Union',
  anyOf: values.map((value) => ({ [Kind]: 'Literal', type: 'string', const: value })),
})
const object = (properties, required = Object.keys(properties)) => ({
  [Kind]: 'Object', type: 'object', properties, required, additionalProperties: false,
})

const readMeta = {
  isReadOnly: true,
  isDestructive: false,
  isConcurrencySafe: true,
  isOpenWorld: false,
  replay: 'safe',
  costHint: { wallMs: 20000 },
  deferLoading: false,
  requiresApproval: 'never',
}
const startMeta = {
  isReadOnly: false,
  isDestructive: false,
  isConcurrencySafe: false,
  isOpenWorld: false,
  replay: 'never',
  costHint: { wallMs: 30000 },
  deferLoading: false,
  requiresApproval: 'always',
}
const stopMeta = { ...startMeta, isDestructive: true }

function bridgeLocation() {
  const root = process.env.VASC_GAUSSIAN_ROOT
  if (!root) throw new Error('VASC_GAUSSIAN_ROOT is not set on the AGH server process.')
  return join(root, 'agh', 'bridge.py')
}

function callBridge(args, signal) {
  const python = process.env.VASC_PYTHON || 'python'
  return new Promise((resolve, reject) => {
    execFile(
      python,
      [bridgeLocation(), ...args],
      { signal, timeout: 30000, windowsHide: true, maxBuffer: 2 * 1024 * 1024 },
      (error, stdout, stderr) => {
        let payload
        try { payload = JSON.parse(stdout) } catch { payload = undefined }
        if (error || !payload?.ok) {
          const detail = payload?.error || { code: 'BRIDGE_FAILURE', message: String(stderr || error?.message || 'Bridge failed').slice(0, 1000) }
          reject(Object.assign(new Error(detail.message), { code: detail.code, details: detail.details }))
          return
        }
        resolve(payload)
      },
    )
  })
}

function result(payload) {
  return { content: [{ type: 'text', text: JSON.stringify(payload, null, 2) }], structured: payload }
}

export const vascGaussianTools = {
  inject: ['extension'],
  apply(ctx) {
    const agnes = ctx.extension()
    agnes.registerTool({
      name: 'vasc_list_profiles',
      description: 'List the committed, allow-listed Vasc-Gaussian experiment profiles. No arbitrary commands or paths are accepted.',
      parameters: object({}, []),
      meta: readMeta,
      async execute(_args, context) { return result(await callBridge(['profiles'], context.signal)) },
    })
    agnes.registerTool({
      name: 'vasc_preflight',
      description: 'Check data, configuration, Python dependencies, CUDA GPU, disk space, and code revision before a Vasc-Gaussian run. Always call this before start.',
      parameters: object({
        profile: string({ pattern: '^[a-z0-9][a-z0-9-]{0,63}$', maxLength: 64 }),
        mode: optional(literalUnion('quick', 'full')),
        gpu_id: optional(string({ pattern: '^[0-9]+$', maxLength: 4 })),
      }, ['profile']),
      meta: readMeta,
      async execute(args, context) {
        context.progress('Checking the Vasc-Gaussian runtime and inputs')
        return result(await callBridge(['preflight', '--profile', args.profile, '--mode', args.mode || 'quick', '--gpu-id', args.gpu_id || '0'], context.signal))
      },
    })
    agnes.registerTool({
      name: 'vasc_start_run',
      description: 'Start an approved Vasc-Gaussian reconstruction pipeline in the background from a committed profile. This writes outputs and consumes GPU time, so every call requires user approval. quick is a connectivity demo; full is the scientific run.',
      parameters: object({
        profile: string({ pattern: '^[a-z0-9][a-z0-9-]{0,63}$', maxLength: 64 }),
        mode: optional(literalUnion('quick', 'full')),
        gpu_id: optional(string({ pattern: '^[0-9]+$', maxLength: 4 })),
      }, ['profile']),
      meta: startMeta,
      async execute(args, context) {
        context.progress('Starting the approved Vasc-Gaussian run')
        const payload = await callBridge(['start', '--profile', args.profile, '--mode', args.mode || 'quick', '--gpu-id', args.gpu_id || '0'], context.signal)
        await agnes.events.append('vasc_run_started', { runId: payload.run_id, profile: args.profile, mode: args.mode || 'quick' })
        return result(payload)
      },
    })
    agnes.registerTool({
      name: 'vasc_run_status',
      description: 'Read current pipeline stage and a bounded log tail for one Vasc-Gaussian run.',
      parameters: object({
        run_id: string({ pattern: '^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$', maxLength: 25 }),
        tail_lines: optional(integer({ minimum: 0, maximum: 100 })),
      }, ['run_id']),
      meta: readMeta,
      async execute(args, context) { return result(await callBridge(['status', '--run-id', args.run_id, '--tail-lines', String(args.tail_lines ?? 40)], context.signal)) },
    })
    agnes.registerTool({
      name: 'vasc_list_runs',
      description: 'List recent Vasc-Gaussian runs and their states.',
      parameters: object({ limit: optional(integer({ minimum: 1, maximum: 50 })) }, []),
      meta: readMeta,
      async execute(args, context) { return result(await callBridge(['runs', '--limit', String(args.limit ?? 10)], context.signal)) },
    })
    agnes.registerTool({
      name: 'vasc_verify_run',
      description: 'Verify that a run completed its declared stages and produced finite 2D, volume, topology, and component metrics. This is execution verification, not a clinical claim.',
      parameters: object({ run_id: string({ pattern: '^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$', maxLength: 25 }) }, ['run_id']),
      meta: readMeta,
      async execute(args, context) { return result(await callBridge(['verify', '--run-id', args.run_id], context.signal)) },
    })
    agnes.registerTool({
      name: 'vasc_compare_runs',
      description: 'Compare the recorded metrics of a baseline run and a candidate run, returning candidate-minus-baseline deltas.',
      parameters: object({
        run_a: string({ pattern: '^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$', maxLength: 25 }),
        run_b: string({ pattern: '^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$', maxLength: 25 }),
      }, ['run_a', 'run_b']),
      meta: readMeta,
      async execute(args, context) { return result(await callBridge(['compare', '--run-a', args.run_a, '--run-b', args.run_b], context.signal)) },
    })
    agnes.registerTool({
      name: 'vasc_stop_run',
      description: 'Stop an active Vasc-Gaussian run and its current stage. This is destructive and always requires user approval.',
      parameters: object({ run_id: string({ pattern: '^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$', maxLength: 25 }) }, ['run_id']),
      meta: stopMeta,
      async execute(args, context) {
        const payload = await callBridge(['stop', '--run-id', args.run_id], context.signal)
        await agnes.events.append('vasc_run_stopped', { runId: args.run_id })
        return result(payload)
      },
    })
  },
}
