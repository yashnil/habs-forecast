export type Colormap = {
  name: string;
  vmin: number;
  vmax: number;
};

export type RegionalEntry = {
  label?: string;
  lat_range?: [number, number];
  median_log_chl?: number;
  tier?: string;
  tier_short?: string;
  pixel_count?: number;
  status?: string;
};

export type Snapshot = {
  schema_version?: number;
  generated_at?: string;
  data_source?: string;
  variable?: string;
  time?: string;
  viewer?: {
    time_headline?: string;
    time_detail?: string;
    what_map_shows?: string;
    composite_note?: string;
  };
  bounds: {
    south: number;
    west: number;
    north: number;
    east: number;
  };
  colormap: Colormap;
  overlay_file?: string;
  land_mask?: string;
  refresh_policy?: {
    recommended_every_days?: number;
    how?: string;
  };
  regional_algae?: {
    reference_percentiles?: { p33: number; p67: number };
    regions: Record<string, RegionalEntry>;
  };
  composite_days_hint?: number;
  /** Optional: filled by export pipeline for audit trail */
  provenance?: {
    checkpoint?: string;
    export_command?: string;
    notes?: string;
  };
};

export type FisheriesContext = {
  version: number;
  disclaimer?: string;
  regions: Record<
    string,
    {
      headline?: string;
      typical_targets?: string;
      operational_note?: string;
      links?: { label: string; url: string }[];
    }
  >;
};

export type HarborFeature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: {
    name: string;
    region_key: string;
  };
};

export type HarborsGeoJSON = {
  type: "FeatureCollection";
  features: HarborFeature[];
};
