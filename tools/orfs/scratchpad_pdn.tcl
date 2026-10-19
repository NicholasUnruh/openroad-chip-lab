# SKY130HD platform grid plus straps for the ORFS sky130ram macro, whose supplies are
# small met3 pads in columns rather than met4 pins. Strap centers are macro-relative
# (R0) and cover the pad columns listed in docs/training-v6-results.md.
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDPE$}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDCE$}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VPWR}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VPB}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^vdd$}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSSE$}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VGND}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VNB}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^gnd$}
global_connect

set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}

define_pdn_grid -name {grid} -voltage_domains {CORE} -pins {met5}
add_pdn_stripe -grid {grid} -layer {met1} -width {0.48} -pitch {5.44} -offset {0} -followpins
add_pdn_stripe -grid {grid} -layer {met4} -width {1.600} -pitch {27.140} -offset {13.570}
add_pdn_stripe -grid {grid} -layer {met5} -width {1.600} -pitch {27.200} -offset {13.600}
add_pdn_connect -grid {grid} -layers {met1 met4}
add_pdn_connect -grid {grid} -layers {met4 met5}

define_pdn_grid -name {scratchpad} -voltage_domains {CORE} -macro \
  -cells {sky130_sram_1rw1r_128x256_8} -orient {R0} -halo {2.0 2.0 2.0 2.0} -grid_over_boundary
# {net center width}; the pitch exceeds the macro width so each call makes one strap.
foreach strap {
  {VSS 160.900 0.690} {VDD 164.105 2.850} {VSS 167.095 0.690} {VDD 168.455 0.690}
  {VSS 214.465 0.690} {VSS 927.045 0.690} {VDD 973.775 0.690} {VSS 975.135 0.690}
  {VDD 978.125 2.850} {VSS 981.330 0.690}
} {
  lassign $strap net center width
  add_pdn_stripe -grid {scratchpad} -layer {met4} -nets $net -width $width -pitch {2000} -offset $center
}
add_pdn_connect -grid {scratchpad} -layers {met3 met4}
add_pdn_connect -grid {scratchpad} -layers {met4 met5}
