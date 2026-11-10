import AXIS from "../../../public/chart/canyon-axis.json";

/**
 * Names of water on the chart (README §5: Literata italic, letter-spaced, water features only), placed where
 * the features are. Positions are geographic facts; zoom ranges keep each name to the scales where its
 * feature reads. "Monterey Canyon" follows the canyon's own thalweg from the CRM (build_chart.py).
 */
export type WaterName = { id: string; name: string; lon: number; lat: number; size: number; minZoom: number; maxZoom?: number; spacing?: number; priority: number };

export const WATER_NAMES: WaterName[] = [
  { id: "pacific-1", name: "Pacific Ocean", lon: -125.4, lat: 37.4, size: 22, minZoom: 4, maxZoom: 7.6, spacing: 0.12, priority: 6 },
  { id: "pacific-2", name: "Pacific Ocean", lon: -124.1, lat: 38.4, size: 22, minZoom: 7.2, maxZoom: 9.2, spacing: 0.12, priority: 6 },
  { id: "pacific-3", name: "Pacific Ocean", lon: -122.2, lat: 36.63, size: 22, minZoom: 9, maxZoom: 12, spacing: 0.12, priority: 6 },
  { id: "pacific-4", name: "Pacific Ocean", lon: -123.05, lat: 37.45, size: 22, minZoom: 9, maxZoom: 12, spacing: 0.12, priority: 6 },
  { id: "pacific-5", name: "Pacific Ocean", lon: -124.55, lat: 40.15, size: 22, minZoom: 9, maxZoom: 12, spacing: 0.12, priority: 6 },
  { id: "pacific-6", name: "Pacific Ocean", lon: -121.25, lat: 35.05, size: 22, minZoom: 9, maxZoom: 12, spacing: 0.12, priority: 6 },
  { id: "pacific-7", name: "Pacific Ocean", lon: -118.95, lat: 33.25, size: 22, minZoom: 9, maxZoom: 12, spacing: 0.12, priority: 6 },
  { id: "monterey-bay", name: "Monterey Bay", lon: -121.98, lat: 36.875, size: 18, minZoom: 8, priority: 9 },
  { id: "carmel-bay", name: "Carmel Bay", lon: -121.955, lat: 36.545, size: 15, minZoom: 11.2, priority: 7 },
  { id: "half-moon-bay", name: "Half Moon Bay", lon: -122.49, lat: 37.47, size: 15, minZoom: 10.5, priority: 7 },
  { id: "sf-bay", name: "San Francisco Bay", lon: -122.3, lat: 37.68, size: 16, minZoom: 9, priority: 8 },
  { id: "farallones", name: "Gulf of the Farallones", lon: -122.78, lat: 37.78, size: 16, minZoom: 8.6, maxZoom: 11.5, priority: 7 },
  { id: "drakes-bay", name: "Drakes Bay", lon: -122.93, lat: 38.0, size: 15, minZoom: 10.6, priority: 7 },
  { id: "bodega-bay", name: "Bodega Bay", lon: -123.03, lat: 38.29, size: 15, minZoom: 10.6, priority: 7 },
  { id: "humboldt-bay", name: "Humboldt Bay", lon: -124.215, lat: 40.765, size: 15, minZoom: 11, priority: 7 },
  { id: "estero-bay", name: "Estero Bay", lon: -120.93, lat: 35.41, size: 15, minZoom: 10.2, priority: 7 },
  { id: "slo-bay", name: "San Luis Obispo Bay", lon: -120.69, lat: 35.12, size: 15, minZoom: 10.6, priority: 7 },
  { id: "sb-channel", name: "Santa Barbara Channel", lon: -119.85, lat: 34.3, size: 16, minZoom: 8, maxZoom: 11.5, priority: 8 },
  { id: "sm-bay", name: "Santa Monica Bay", lon: -118.62, lat: 33.88, size: 16, minZoom: 9, priority: 8 },
  { id: "sp-channel", name: "San Pedro Channel", lon: -118.42, lat: 33.53, size: 16, minZoom: 9, priority: 7 },
  { id: "sd-bay", name: "San Diego Bay", lon: -117.18, lat: 32.665, size: 15, minZoom: 11.2, priority: 7 },
];

/** The canyon's axis, west from its head off Moss Landing: [lon, lat, depth m]. */
export const CANYON_AXIS = AXIS as [number, number, number][];
export const CANYON_NAME = { id: "monterey-canyon", name: "Monterey Canyon", size: 17, minZoom: 9.4, priority: 8 };
