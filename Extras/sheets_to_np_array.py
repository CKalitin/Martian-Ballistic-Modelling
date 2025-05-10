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
        v = int(parts[0])
        h = int(parts[1])
        velocities.append(v)
        alts.append(h)
    
    # Convert to numpy arrays
    v_arr = np.array(velocities, dtype=int)
    h_arr = np.array(alts,       dtype=int)
    
    # Sort by velocity descending
    idx = np.argsort(-v_arr)
    return v_arr[idx], h_arr[idx]

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
    lines.append(f"{name.lower()}_vel_alt = [")
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
5,537	123,770
5,537	121,311
5,537	118,852
5,537	116,667
5,548	84,426
5,548	86,885
5,548	89,071
5,548	91,530
5,548	93,989
5,548	96,175
5,548	98,634
5,548	101,093
5,548	103,279
5,548	105,738
5,548	108,197
5,537	110,383
5,537	112,842
5,537	115,027
5,548	80,055
5,548	79,508
5,548	79,508
5,548	80,055
5,548	75,410
5,548	72,951
5,537	70,492
5,537	68,033
5,526	65,574
5,503	63,115
5,481	60,929
5,459	58,743
5,415	56,557
5,370	54,372
5,315	52,459
5,315	52,459
5,337	52,732
5,237	50,273
5,160	48,634
5,082	47,268
5,004	45,902
4,926	44,809
4,849	43,989
4,760	42,896
4,671	42,077
4,583	41,257
4,505	40,710
4,416	39,891
4,327	39,071
4,239	38,525
4,150	37,705
4,061	37,158
3,961	36,612
3,872	36,066
3,784	35,519
3,695	35,246
3,595	34,699
3,506	34,153
3,417	33,607
3,318	33,060
3,229	32,787
3,229	32,787
3,318	33,060
3,417	33,607
3,506	34,153
3,595	34,699
3,695	35,246
3,784	35,519
3,872	36,066
3,961	36,612
4,061	37,158
4,150	37,705
4,239	38,525
4,327	39,071
4,416	39,891
4,505	40,710
4,583	41,257
4,671	42,077
4,760	42,896
4,849	43,989
4,926	44,809
5,004	45,902
5,082	47,268
5,160	48,634
5,237	50,273
2,752	30,055
2,663	29,508
2,574	29,235
2,474	28,962
2,386	28,415
2,297	28,142
2,208	27,596
2,108	27,322
2,019	26,776
1,931	26,230
1,842	25,683
1,753	25,137
1,664	24,863
1,564	24,317
1,487	23,770
1,398	22,951
1,309	22,404
1,221	21,858
1,132	21,038
1,043	20,219
965	19,399
877	18,579
799	17,486
721	16,667
644	15,301
566	13,934
499	12,295
422	10,656
333	10,109
333	10,109
422	10,656
499	12,295
566	13,934
644	15,301
721	16,667
799	17,486
877	18,579
965	19,399
1,043	20,219
1,132	21,038
1,221	21,858
1,309	22,404
1,398	22,951
1,487	23,770
1,564	24,317
1,664	24,863
1,753	25,137
1,842	25,683
1,931	26,230
2,019	26,776
2,108	27,322
2,208	27,596
2,297	28,142
2,386	28,415
2,474	28,962
2,574	29,235
2,663	29,508
2,752	30,055
100	7,923
67	4,372
44	546
    """
    
vel,alt = parse_vel_alt(raw_data)
print(format_for_python("Pheonix", vel, alt))