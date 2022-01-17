import common
import csv


common.log("Hist")

while True:
    n_changed = 0
    for file in common.data_files_iter(True):
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        last_rows = []
        changed = False
        for row in data:
            for key in ["o","c","h","l"]:
                row[key] = common.price_to_float(row[key])
            if len(last_rows) == 3:
                if abs(last_rows[2]["c"] - 2 * last_rows[1]["c"] + last_rows[0]["c"]) / sum(last_rows[i]["c"] for i in range(3)) * 3 > 1.5:
                    common.log(file)
                    common.log(last_rows)
                    for key in ["o","c","h","l"]:
                        last_rows[1][key] = round((last_rows[0][key] + last_rows[2][key]) / 2, 4)
                    changed = True
                    n_changed += 1
                    common.log(last_rows)
            last_rows.append(row)
            if len(last_rows) > 3:
                last_rows = last_rows[1:]
        if changed:
            with open(file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    common.log("n_changed:", n_changed)
    if n_changed == 0:
        break

common.log("Real time")

while True:
    n_changed = 0
    for file in ["data/real_time/2021/09/16.csv"]: #common.data_files_iter(False):
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        last_comp_rows = {}
        changed = False
        for row in data:
            conidex = row["conidex"]
            if conidex not in last_comp_rows:
                last_comp_rows[conidex] = []
            last_rows = last_comp_rows[conidex]
            for key in ["31"]:
                row[key] = common.price_to_float(row[key])
            if len(last_rows) == 3:
                if abs(last_rows[2]["31"] - 2 * last_rows[1]["31"] + last_rows[0]["31"]) / sum(last_rows[i]["31"] for i in range(3)) * 3 > 1.5:
                    common.log(file)
                    common.log(last_rows)
                    for key in ["31"]:
                        last_rows[1][key] = round((last_rows[0][key] + last_rows[2][key]) / 2, 4)
                    changed = True
                    n_changed += 1
                    common.log(last_rows)
            last_rows.append(row)
            if len(last_rows) > 3:
                last_rows = last_rows[1:]
        if changed:
            with open(file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    common.log("n_changed:", n_changed)
    if n_changed == 0:
        break
