import datetime as dt

s = str(dt.datetime.now())
time = ''.join(x for x in s if x.isdigit())
file_name = f"v1 - depression\models\model_{time[2:13]}.py"

f = open(file_name, "x")
