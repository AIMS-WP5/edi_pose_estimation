import sys

file_path = sys.argv[1]
with open(file_path, "r") as file:
    no_detections = 0
    detections = 0
    total_time = 0
    total_time_sum = 0
    first = True
    for row in file:
        if "(no detections)" in row:
            no_detections += 1
            continue
        detections += 1
        time = float(row.split()[-1][:-2])
        if first:
            total_time = time
        else:
            total_time += time
            total_time_sum += total_time
            # print(f"Total time: {total_time:.2f} ms")
        first = not first

    detections = detections//2
    print(f"Detected: {detections}/{detections + no_detections} {100 * detections/(detections + no_detections):.2f}%")
    print(f"Average detection time per frame: {total_time_sum/detections:.2f} ms")
