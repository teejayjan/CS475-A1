#!/bin/csh

#number of threads:
foreach t (1 2 4 8 16)
    echo NUMT = $t
    foreach s (1 10 100 1000 10000 100000 500000)
        echo NUMTRIALS = $s
        g++-10 -DNUMT=$t -DNUMTRIALS=$s proj1.cpp -o proj1 -lm -fopenmp
        ./proj1
    end
end