### Instructions to run CCS Example 

1. Start up your python virtual env if using one 
2. Boot up the server through the command: python -m charmrun.start +p<N> ++server ++local ccs_server.py 
3. Grab and store the the Server IP and Server Port that will appear in stdout when you run the above command 
4. In another terminal, cd to the converse/ccstest example in the charm++ repository and run a client program through the command: ./client server_ip server_port 
