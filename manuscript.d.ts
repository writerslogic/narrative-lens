/** Input defaults are optional; result contracts are generated from Rust. */
import type * as Core from './manuscript.generated'
export * from './manuscript.generated'
export type ResolvedConfiguration = Core.Configuration
export type Thresholds = Partial<Core.Thresholds>
export type AnalysisLimits = Partial<Core.AnalysisLimits>
export type Suppression = Pick<Core.Suppression, 'reason'> & Partial<Omit<Core.Suppression, 'reason'>>
export type Configuration = Partial<Omit<Core.Configuration, 'thresholds' | 'limits' | 'suppressions'>> & {
  thresholds?: Thresholds
  limits?: AnalysisLimits
  suppressions?: Suppression[]
}
export type Scene = Pick<Core.Scene, 'id' | 'text'> & Partial<Omit<Core.Scene, 'id' | 'text'>>
export type Manuscript = Pick<Core.Manuscript, 'documentId'> & {
  scenes: Scene[]
  knownCharacters?: string[]
  configuration?: Configuration
}
export type RuleDefinition = Omit<Core.RuleDefinition, 'id'> & { id: Core.RuleId }
