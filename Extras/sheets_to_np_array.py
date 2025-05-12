# This converts the Google Sheets velocity, altitude columns into the numpy array required

import numpy as np

def parse_vel_alt(data_str):
    """
    Parse a tab‑or‑space‑delimited multiline string with header
    Velocity (m/s)   Altitude (m)
    into two sorted numpy arrays (velocities desc, same order altitudes).
    """
    lines = data_str.strip().splitlines()
    # Skip header
    data_lines = [l for l in lines if not l.lower().startswith('velocity')]
    
    velocities = []
    alts       = []
    for line in data_lines:
        # split on whitespace or tabs
        parts = line.strip().replace(',', '').split()
        if len(parts) < 2:
            continue
        v = int(float(parts[0]))
        h = int(float(parts[1]))
        velocities.append(v)
        alts.append(h)
    
    # Convert to numpy arrays
    v_arr = np.array(velocities, dtype=int)
    h_arr = np.array(alts,       dtype=int)
    
    return v_arr, h_arr

def format_for_python(name, v_arr, h_arr):
    """
    Return a formatted Python snippet defining:
      <name>_vel_alt = [
        np.array([...]),
        np.array([...]),
        "<Name>"
      ]
    """
    lines = []
    lines.append(f"# {name} Entry:")
    lines.append(f"{name.lower()} = [")
    # velocities
    vel_list = ", ".join(str(x) for x in v_arr)
    lines.append("    # Velocity (m/s)")
    lines.append("    np.array([")
    lines.append("    " + vel_list)
    lines.append("    ]),")
    # altitudes
    alt_list = ", ".join(str(x) for x in h_arr)
    lines.append("    # Altitude (m)")
    lines.append("    np.array([")
    lines.append("    " + alt_list)
    lines.append("    ]),")
    # name
    lines.append(f'    "{name}"')
    lines.append("]")
    return "\n".join(lines)

if __name__ == "__main__":
    # paste your sheet output into this triple‑quoted string:
    raw_data = """Velocity (m/s)	Altitude (m)
0	-16.089423
37.134272	-13.976471
50.079256	-13.129412
72.732977	-11.223529
80.823592	-9.529412
87.296084	-7.623529
93.768576	-5.717647
98.622945	-4.235294
103.477314	-2.964706
111.567929	-1.905882
119.658544	-1.058824
129.367282	-0.423529
142.312265	-0.211765
158.493495	-0.423529
174.674725	-1.694118
189.237832	-3.811765
198.94657	-5.717647
211.891553	-8.470588
224.836537	-12.070588
234.545275	-16.729412
239.399644	-18.423529
241.017767	-20.752941
245.872136	-22.447059
249.108382	-25.2
250.726505	-27.741176
253.962751	-30.070588
257.198997	-31.764706
260.435243	-33.247059
263.671489	-35.788235
266.907735	-40.023529
270.143981	-45.105882
273.380227	-49.552941
276.616472	-53.576471
281.470841	-60.352941
286.32521	-66.070588
287.943333	-67.764706
289.561456	-69.247059
294.415825	-70.941176
299.270194	-72.635294
304.124563	-73.482353
310.597055	-72.847059
315.451424	-73.694118
321.923916	-74.541176
325.160162	-76.658824
328.396408	-78.776471
333.250777	-81.105882
336.487023	-81.529412
342.959515	-80.894118
347.813883	-81.317647
352.668252	-82.164706
355.904498	-81.529412
357.522621	-77.929412
359.140744	-74.541176
362.37699	-64.8
365.613236	-57.811765
367.231359	-53.788235
370.467605	-52.941176
373.703851	-54.847059
375.321974	-60.776471
376.940097	-65.223529
378.55822	-71.364706
380.176343	-79.623529
381.794466	-86.4
383.412589	-90
393.121327	-90
394.73945	-83.223529
401.211942	-88.941176
404.448188	-80.682353
410.92068	-75.6
412.538803	-63.317647
    """
    
vel,alt = parse_vel_alt(raw_data)
print(format_for_python("Perseverance", vel, alt))