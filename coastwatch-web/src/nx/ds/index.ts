/**
 * Neritic component library. v14 synthesis: docs/coastwatch/v14/design-system/README.md (tokens, Ticket, Dock, Pass,
 * Sheet, buttons, chips, tabs, the rolling figure, the map layer API, motion). Earlier: v12 README section 9; reference in
 * components.md; living catalog at /system). v13 additions (layout, controls, states, the plot kit):
 * docs/coastwatch/v13/design-system/README.md. Import tokens.css once at the app root and put `fontVariables`
 * (./fonts) on <html>. fonts.ts is not re-exported here: it calls next/font, which belongs in a layout.
 */
export * from "./motion";
export * from "./format";
export * from "./kinds";
export * from "./mode";
export { ChartModeRoot, usePortalContainer } from "./Root";
export { cx, svgId, VisuallyHidden } from "./util";
export { useMedia, usePhone, useElementWidth, PHONE_QUERY } from "./useMedia";
export { Icon, ICON_NAMES, type IconName, type IconProps } from "./Icon";
export { Button, type ButtonProps, type ButtonVariant, type ButtonSize, type ButtonGlyph } from "./Button";
export { ButtonGroup, type ButtonGroupProps } from "./ButtonGroup";
export { IconButton, type IconButtonProps } from "./IconButton";
export { RangeControl, ScrubNumber, Disclosure, ToggleRow, Hint, type RangeControlProps, type ScrubNumberProps, type DisclosureProps, type ToggleRowProps, type HintProps } from "./Controls";
export { Drawer, EditableText, OptionList, Inspector, Facts, type DrawerProps, type EditableTextProps, type OptionListProps, type OptionItem, type InspectorProps } from "./Panels";
export { GateList, Advisories, type GateListProps, type GateItem, type GateFactItem, type GateItemState, type AdvisoriesProps, type AdvisoryItem } from "./GateList";
export { Figure, type FigureProps, type FigureFormat } from "./Figure";
export { useSettledNumber } from "./useSettledNumber";
export { DataState, DataStateMark, DataStateLegend, Availability, Skeleton, Recovered, DATA_STATE_KINDS, DATA_STATE_WORDS, type DataStateKind, type DataStateProps, type AvailabilityProps, type AvailabilityRow, type AvailabilitySpan, type SkeletonProps, type RecoveredProps } from "./DataState";
export * from "./layout";
export * from "./plot";
export { Mark, Legend, hatchSegments, type MarkProps, type LegendProps } from "./Mark";
export { Statement, RegulationRule, type StatementProps } from "./Statement";
export { BigDate, type BigDateProps, type DateSpan } from "./BigDate";
export { Agenda, agendaPhase, type AgendaItem, type AgendaProps, type AgendaPhase } from "./Agenda";
export { Stamp, ProvenanceSheet, stampText, type Provenance, type StampProps } from "./Stamp";
export { Dialog, Popover, Tooltip, TooltipProvider, type DialogProps, type PopoverProps, type TooltipProps } from "./Overlay";
export { Margin, MarginRows, MarginRow, type MarginProps, type MarginRowProps, type MarginRowState } from "./Margin";
export { MarginSheet, type MarginSheetProps } from "./MarginSheet";
export { PortFinder, type PortEntry, type PortFinderProps } from "./PortFinder";
export { Unavailable, Stale, Offline, SourcesDiffer, Loading, type SourceClaim } from "./States";
export { Sounding, type SoundingProps } from "./chart/Sounding";
export { ChartNote, PortGlyph, type ChartNoteProps, type ChartNoteVariant } from "./chart/ChartNote";
export { RampLegend, rampPosition, type RampLegendProps } from "./chart/RampLegend";
export { Readout, type ReadoutProps, type ReadoutState } from "./chart/Readout";
export { anchorTranslate, anchorAlign, byPriority, type Anchor } from "./chart/place";
export * from "./series/series";
export { SeriesChart, type SeriesChartProps } from "./series/SeriesChart";
export { SmallMultiples, type Strip, type SmallMultiplesProps } from "./series/SmallMultiples";
export { Brush, clampWindow, frameEdges, MIN_FRAME, type BrushProps, type BrushHandle } from "./series/Brush";
export { Field, Stepper, Segmented, Toggle, type FieldProps, type StepperProps, type SegmentedProps, type ToggleProps } from "./worksheet/Inputs";
export { Waterfall, SensitivityTable, type WaterfallProps, type SensitivityTableProps } from "./worksheet/Waterfall";
export { layoutWaterfall, type WaterfallStep, type WaterfallLayout, type Bar, type StepKind } from "./worksheet/waterfallLayout";

/* ---------------------------------------------------------------- v14 synthesis (docs/coastwatch/v14/design-system/README.md) */
export { Ticket, TicketMain, TicketStub, type TicketProps, type Place } from "./stage/Ticket";
export { Dock, Pass, type DockProps, type PassProps } from "./stage/Dock";
export { Sheet, setSheetRest, useSheetRest } from "./stage/Sheet";
export { Glyph, Micro, StampLine, Slot, Pennant, WindArrow, LimitGauge, GLYPH_WORDS, type GlyphKind } from "./synth/Grammar";
export { FollowButton, type FollowButtonProps } from "./synth/FollowButton";
export { Chip, LayerChips, type ChipProps } from "./synth/Chips";
export { Tabs, TabPanel, tabIds, type TabsProps, type TabItem } from "./synth/Tabs";
export { RollingNumber, type RollingNumberProps } from "./synth/RollingNumber";
export { MapLayers, useMapLayers, useMapLayer, setMapLayer, toggleMapLayer, MAP_LAYERS, type MapLayerId, type MapLayerDef } from "@/nx/chart/mapLayers";
