const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { WASI } = require('node:wasi');

async function main() {
  if (process.argv.length !== 5 || !['upstream', 'adapted'].includes(process.argv[4])) {
    throw new Error('Usage: node tests/component_real_model_wasi.cjs <component_real_model.wasm> <ignored-target-cache-dir> upstream|adapted');
  }
  const artifact = resolve(process.argv[2]);
  const cache = resolve(process.argv[3]);
  const wasi = new WASI({
    version: 'preview1', returnOnExit: true,
    args: [artifact, cache, process.argv[4]],
    preopens: { [cache]: cache },
  });
  const module = await WebAssembly.compile(readFileSync(artifact));
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  process.exitCode = wasi.start(instance);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
