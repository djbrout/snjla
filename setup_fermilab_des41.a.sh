source /cvmfs/des.opensciencegrid.org/eeups/startupcachejob21i.sh
kx509
voms-proxy-init -rfc -noregen -valid 24:00 -voms des:/des/Role=Analysis

setup Y2Nstack 1.0.6+18
setup diffimg gwdevel8
setup ftools v6.17
setup autoscan v3.1+0
setup easyaccess
setup extralibs 1.0
setup ifdhc
echo "EUPS setup complete"

source /cvmfs/fermilab.opensciencegrid.org/products/common/etc/setups
setup jobsub_client
echo 'done setting up jobsub_client'