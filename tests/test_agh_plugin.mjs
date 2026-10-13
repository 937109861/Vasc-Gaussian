import assert from 'node:assert/strict'
import { vascGaussianTools } from '../agh/vasc-tools/index.mjs'

const tools = []
const events = []
const extension = {
  registerTool(definition) {
    tools.push(definition)
    return () => {}
  },
  events: {
    async append(name, data) { events.push({ name, data }) },
  },
}
const context = { extension: () => extension }

vascGaussianTools.apply(context)

assert.deepEqual(
  tools.map((tool) => tool.name),
  [
    'vasc_list_profiles',
    'vasc_preflight',
    'vasc_start_run',
    'vasc_run_status',
    'vasc_list_runs',
    'vasc_verify_run',
    'vasc_compare_runs',
    'vasc_stop_run',
  ],
)
for (const tool of tools) {
  assert.equal(tool.parameters[Symbol.for('TypeBox.Kind')], 'Object')
  assert.equal(typeof tool.execute, 'function')
  for (const key of [
    'isReadOnly', 'isDestructive', 'isConcurrencySafe', 'isOpenWorld',
    'replay', 'costHint', 'deferLoading', 'requiresApproval',
  ]) assert.ok(Object.hasOwn(tool.meta, key), `${tool.name} is missing meta.${key}`)
}
assert.equal(tools.find((tool) => tool.name === 'vasc_start_run').meta.requiresApproval, 'always')
assert.equal(tools.find((tool) => tool.name === 'vasc_stop_run').meta.isDestructive, true)

console.log(`Validated ${tools.length} AGH tool definitions.`)
