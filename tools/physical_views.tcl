# Heatmaps, clock-net view and per-endpoint slack of a routed design; run by tools/physical_views.sh.
# Environment: ODB, SDC, SPEF, LIBS (space-separated liberty files), OUT (output directory).
foreach lib $::env(LIBS) { read_liberty $lib }
read_db $::env(ODB)
read_sdc $::env(SDC)
read_spef $::env(SPEF)
set_propagated_clock [all_clocks]
set out $::env(OUT)

# Every setup and hold endpoint with its slack, for the histogram in tools/plot_figures.py.
report_checks -path_delay max -group_path_count 1000000 -endpoint_path_count 1 \
  -unique_paths_to_endpoint -format end > $out/endpoints_setup.txt
report_checks -path_delay min -group_path_count 1000000 -endpoint_path_count 1 \
  -unique_paths_to_endpoint -format end > $out/endpoints_hold.txt

set block [ord::get_db_block]
set die [$block getDieArea]
set area [list [ord::dbu_to_microns [$die xMin]] [ord::dbu_to_microns [$die yMin]] \
  [ord::dbu_to_microns [$die xMax]] [ord::dbu_to_microns [$die yMax]]]
set um_per_px 2.0

proc view {name} {
  global out area um_per_px
  save_image -area $area -resolution $um_per_px $out/$name.png
}

gui::fit
gui::set_display_controls "*" visible false
gui::set_display_controls "Instances/*" visible true
gui::set_display_controls "Instances/Physical/*" visible false

gui::set_display_controls "Heat Maps/Placement Density" visible true
gui::set_heatmap Placement ShowLegend 1
view density
gui::set_display_controls "Heat Maps/Placement Density" visible false

# The final database holds no global-route guides; the GUI reruns global routing for this map.
gui::set_display_controls "Heat Maps/Routing Congestion" visible true
gui::set_heatmap Routing ShowLegend 1
view congestion
gui::set_display_controls "Heat Maps/Routing Congestion" visible false

# Clock tree: clock nets on every layer, clock buffers/inverters and the macro outline.
gui::set_display_controls "Instances/*" visible false
gui::set_display_controls "Instances/StdCells/Clock tree/*" visible true
gui::set_display_controls "Instances/Macro" visible true
gui::set_display_controls "Layers/*" visible true
gui::set_display_controls "Shape Types/Routing/*" visible true
gui::set_display_controls "Nets/Clock" visible true
view clock
exit
