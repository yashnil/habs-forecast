"use client";
import { Button } from "../Button";
import { useChartPane } from "./pane";
import s from "./Desk.module.css";

export interface ChartToggleProps {
  className?: string;
}

/**
 * The desk header's chart control: shows the chart where it is not beside the desk (the overlay pane on a
 * narrow or short screen) and collapses it to the rail and back where it is. Hidden on a phone, where the chart
 * strip has its own control.
 */
export function ChartToggle({ className }: ChartToggleProps) {
  const pane = useChartPane();
  const shown = pane.open || !pane.collapsed;
  return (
    <span className={s.chartToggle} data-open={pane.open || undefined} data-collapsed={pane.collapsed || undefined}>
      <Button variant="secondary" icon="layers" onClick={pane.toggle} aria-pressed={shown} className={className} data-testid="nx-chart-toggle">
        Chart
      </Button>
    </span>
  );
}
