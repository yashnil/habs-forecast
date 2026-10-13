/** The subset of the published CoastWatch data contract (schemas/) that the prototypes read. */
export interface GridSpec {
  url: string;
  encoding: "uint16le+gzip";
  width: number;
  height: number;
  lat_first: number;
  lat_step: number;
  lon_first: number;
  lon_step: number;
  scale_factor: number;
  add_offset: number;
  nodata: number;
  transform: "none" | "log10";
  chunks: null | { rows: number; cols: number; url_template: string; present: string[] };
}
export interface PaletteStop { value: number; color: string }
export interface Layer {
  layer_id: string;
  group_id: string;
  product_class: "official_forecast" | "observation" | string;
  title: string;
  short_title: string;
  variable: string;
  units: string;
  threshold_text?: string | null;
  description: string;
  resolution_deg?: number | null;
  native_resolution_m?: number | null;
  time: { issued_date: string | null; valid_date: string | null; valid_time?: string | null; observed_date?: string | null; lead_days?: number | null; observed_times?: string[] };
  image: null | { url: string; corners_lnglat: [number, number][]; bounds_lnglat: number[] };
  grid: GridSpec | null;
  tiles: null | { url_template: string; tile_size: number; max_native_zoom: number; min_zoom?: number; bounds_lnglat: number[] };
  palette: null | { id: string; domain: [number, number]; stops: PaletteStop[]; interpolation: "step" | "linear"; scale: string };
  caveats: string[];
  vectors?: null | { u_grid: GridSpec; v_grid: GridSpec; speed_max: number };
  coverage?: unknown;
}
export interface SourceStatus {
  source_id: string;
  title: string;
  product_class: string;
  last_attempt_at: string | null;
  last_success_at: string | null;
  outcome: "ok" | "failed" | string;
  error?: string | null;
}
export interface Manifest {
  generated_at: string;
  pipeline_run_id: string;
  layers: Layer[];
  forecast_runs: { group_id: string; issued_date: string; leads_available: number[]; leads_missing: number[] }[];
  sources: SourceStatus[];
  ports_url: string;
  official_url: string;
  port_intel_url: string;
  observations_url: string;
  fisheries_url: string;
}

export interface OfficialRecord {
  id: string;
  agency: "CDPH" | "CDFW" | string;
  action: string;
  status: string;
  title: string;
  summary: string;
  official_text?: string;
  fishery: string;
  species: string[];
  area: { type: string; description: string; counties?: string[]; lat_north?: number; lat_south?: number; geometry_note?: string };
  effective_date: string | null;
  lifted_date: string | null;
  sources: { label: string; url: string; kind: string; published_date: string | null }[];
}
export interface Official {
  generated_at: string;
  registry: { records: OfficialRecord[]; review?: unknown; hotlines?: unknown };
  watch: { source_id: string; url: string; checked_at: string; ok: boolean; new_items?: string[] }[];
  conflicts: string[];
  geometry: GeoJSON.FeatureCollection;
  policy: { verified_max_age_days: number; aging_max_age_days: number; note: string };
}

export interface Station {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  region: string;
  nearest_port_name: string;
  nearest_port_km: number;
  status: string;
  sample_times: string[];
  series: { variable: string; values: (number | null)[]; qualifiers: Record<string, string> }[];
  summaries: { variable: string; n_measured: number; last_date: string | null; last_value: number | null; days_since_last: number | null; n_last_365d: number; median_interval_days_365d: number | null; max_last_365d: number | null }[];
  source_url: string;
}
export interface Observations {
  generated_at: string;
  program: string;
  variables: { id: string; label: string; units: string; kind: string }[];
  stations: Station[];
  caveats: string[];
}
export interface Port {
  port_code: number;
  name: string;
  display_name: string;
  county: string;
  region: string;
  lon: number;
  lat: number;
}
