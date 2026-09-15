// Type-only check for the generated TypeScript definitions (index.d.ts), run
// via `npm run test:types` (tsc --noEmit). This file has no runtime
// assertions -- a type error here is the test failure.

import {
  computeReadability,
  checkGrammar,
  listStructureTemplates,
  type ReadabilityResultJs,
  type GrammarFindingJs,
} from '../../index.js';

const readability: ReadabilityResultJs = computeReadability('Example passage.');
const fkgl: number = readability.fkgl;
const gradeLabel: string = readability.gradeLevels.fkgl;

const findings: GrammarFindingJs[] = checkGrammar('Example passage.');
const firstKind: string | undefined = findings[0]?.kind;

const templates: string[] = listStructureTemplates();

void fkgl;
void gradeLabel;
void firstKind;
void templates;
