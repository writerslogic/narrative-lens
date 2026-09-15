// ESM wrapper around the CommonJS native-addon loader (index.js, auto-generated
// by `napi build --platform`). A native addon loads via `require`, so this
// file is hand-maintained, not generated -- keep its exports in sync with
// index.d.ts when adding a new `#[napi]` binding.

import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const native = require('./index.js')

export const { computeReadability, checkGrammar, listStructureTemplates } = native
export default native
