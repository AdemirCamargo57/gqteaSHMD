#
# +------------------------------------------------------------+
# |                    Fit plotting script                     |
# +------------------------------------------------------------+
#
#
# *** Definition of the kinetic model: ***
#     |  S0   S1 
#-----+----------
#  S0 |   .    . 
#  S1 | k10    . 
#
#(Initial species: rows; Final species: columns)
#
#
# *** Species and initial value: ***
# S0        0
# S1        S1__0
#
#
# *** Reaction rates: ***
# k10
#
#


# ========================================================
#<<
# *** Gnuplot general options: ***
set xlabel "Global Fit Time Axis (fs)"
set ylabel "Population"
set xrange [0:5.00]
set yrange [0:1]
unset key
set tmargin 1
set bmargin 4
set lmargin 8
set rmargin 22

# *** Label with time constants: ***
set label 1 "t(k10) = 104172.5 fs\np0(S1) =     1.0\n" at 5.50,0.8 left

# *** Label with groups: ***
set label 2 "S1\n$3+$4" at 2.50,0.95 center
p "fit_results.txt" u 1:2 w p pt 7 ps 0.4 lc rgb "black", "" u 1:3 w l lw 2 lc rgb "red"

pause -1

# *** Infos: ***
# user@host (scrubbed)
# Date: 2026-11-14 08:37:52.063046
# Current directory: (scrubbed)

