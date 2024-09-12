How to run the sample code

# One producer
./nvscistream_multicast_sample -n 3 -p nvscistream_0 nvscistream_2 nvscistream_4 &

# Three consumers
./nvscistream_multicast_sample -c nvscistream_1 -q 1 &
./nvscistream_multicast_sample -c nvscistream_3 -q 1 &
./nvscistream_multicast_sample -c nvscistream_5 -q 1 &
