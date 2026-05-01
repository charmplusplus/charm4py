#!/usr/bin/env python3
import os
import sys
import csv
import statistics

if len(sys.argv) != 4:
  print('Please use', sys.argv[0], '[job name] [start node count] [end node count]')
  exit()

job_name = sys.argv[1]
start_node_count = int(sys.argv[2])
end_node_count = int(sys.argv[3])

csv_filename = job_name + '.csv'
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Number of GPUs', 'Charm4py-H-Total', 'error', 'Charm4py-H-Comm', 'error', 'Charm4py-D-Total', 'error', 'Charm4py-D-Comm', 'error'])

def is_host(index):
  return index % 6 == 0 or index % 6 == 1 or index % 6 == 2

node_count_list = []
cur_node_count = start_node_count
while cur_node_count <= end_node_count:
  node_count_list.append(cur_node_count)
  cur_node_count *= 2

for node_count in node_count_list:
  print('Node count:', str(node_count))
  total_str = 'grep -ir "Average time per" ' + job_name + '-n' + str(node_count) + '.* | cut -d " " -f5'
  comm_str = 'grep -ir "Communication time" ' + job_name + '-n' + str(node_count) + '.* | cut -d " " -f5'

  total_stream = os.popen(total_str)
  total_lines = total_stream.readlines()
  total_times = list(map(lambda x: x, list(map(float, list(map(str.rstrip, total_lines))))))
  comm_stream = os.popen(comm_str)
  comm_lines = comm_stream.readlines()
  comm_times = list(map(lambda x: x, list(map(float, list(map(str.rstrip, comm_lines))))))

  h_total_times = [total_times[i] for i in range(len(total_times)) if is_host(i)]
  h_comm_times = [comm_times[i] for i in range(len(comm_times)) if is_host(i)]
  d_total_times = [total_times[i] for i in range(len(total_times)) if not is_host(i)]
  d_comm_times = [comm_times[i] for i in range(len(comm_times)) if not is_host(i)]
  print('H total:', h_total_times)
  print('H comm:', h_comm_times)
  print('D total:', d_total_times)
  print('D comm:', d_comm_times)

  h_total_avg = round(statistics.mean(h_total_times), 2)
  h_total_stdev = round(statistics.stdev(h_total_times), 2)
  h_comm_avg = round(statistics.mean(h_comm_times), 2)
  h_comm_stdev = round(statistics.stdev(h_comm_times), 2)
  d_total_avg = round(statistics.mean(d_total_times), 2)
  d_total_stdev = round(statistics.stdev(d_total_times), 2)
  d_comm_avg = round(statistics.mean(d_comm_times), 2)
  d_comm_stdev = round(statistics.stdev(d_comm_times), 2)
  print('H total avg:', h_total_avg, 'stdev:', h_total_stdev)
  print('H comm avg:', h_comm_avg, 'stdev:', h_comm_stdev)
  print('D total avg:', d_total_avg, 'stdev:', d_total_stdev)
  print('D comm avg:', d_comm_avg, 'stdev:', d_comm_stdev)

  writer.writerow([str(node_count), str(h_total_avg), str(h_total_stdev), str(h_comm_avg), str(h_comm_stdev), str(d_total_avg), str(d_total_stdev), str(d_comm_avg), str(d_comm_stdev)])
