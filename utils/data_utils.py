import os

# FILES
source_filename = os.getcwd() + '/data/b1378_sampled.txt'
dest_filename = os.getcwd() + '/data/b1378_downsampled_shortened.txt'

# FILTER VALUES
t_start = 2462.0
t_end = 7462.0
downsampling_rate = 100


# READ SOURCE FILE
s = open(source_filename, 'r')
lines = list()
for i, line in enumerate(s):
    # WRITE HEADERS
    if i < 2:
        lines.append(line)
        continue

    # FILTERING
    columns = line.split()
    if (
        i % downsampling_rate == 0
        and float(columns[6]) > t_start
        and float(columns[6]) < t_end
    ):
        lines.append(line)
s.close()


# WRITE TO DEST
d = open(dest_filename, 'w')
d.writelines(lines)
d.close()