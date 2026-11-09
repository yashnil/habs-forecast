import type { DerivedRaster, ImageRaster, TileRaster } from "@/components/map/MapCanvas";
import type { LayerArtifact, PortsCollection } from "@/generated/schema";

export type StageField = "chlorophyll" | "charm" | "outlook" | "sst" | "none";
/** Which family a field belongs to; it is said in words on the caption. */
export type StageKind = "observed" | "model" | "outlook" | "analysis";

/** The caption's content: what · kind · date, and its ramp. The caveat for the field is added by the stage itself. */
export type StageCaption = {
  what: string;
  kind: StageKind;
  /** the field's own date, in words ("Oct 7", "Oct 3–12") */
  date: string;
  /** a model run whose days are all past steps back: said beside the kind */
  past?: boolean;
  ramp?: { css: string; labels: string[]; unit?: string } | null;
};

export type StageFieldData<R> = { raster: R; caption: StageCaption };

/** Everything the stage draws, gathered on the server (loadStageData) and small enough to pass as a prop. */
export type StageData = {
  baseUrl: string;
  ports: PortsCollection | null;
  official: GeoJSON.FeatureCollection | null;
  chlorophyll: StageFieldData<TileRaster> | null;
  charm: StageFieldData<ImageRaster> | null;
  outlook: StageFieldData<DerivedRaster> | null;
  sst: StageFieldData<DerivedRaster> | null;
  /** The measured-currents hour shown, when the feed has one: only what loading its u/v grids needs */
  currents: { layer: Pick<LayerArtifact, "layer_id" | "vectors">; hour: string } | null;
};

/** A mark on the stage. `kind` picks its family colour; never a data value. */
export type StageMark = { id: string; lon: number; lat: number; kind: "port" | "site" | "buoy"; label?: string };
