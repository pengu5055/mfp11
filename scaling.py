import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "BigBlueTerm437 Nerd Font Mono"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'BigBlueTerm437 Nerd Font Mono'
plt.rcParams['mathtext.it'] = 'BigBlueTerm437 Nerd Font Mono:italic'
plt.rcParams['mathtext.bf'] = 'BigBlueTerm437 Nerd Font Mono:bold'
plt.rcParams['mathtext.sf'] = 'BigBlueTerm437 Nerd Font Mono'
plt.rcParams.update({'axes.unicode_minus' : False})

data = {'m=1, n=1': (0.7205061947899575, 18684800), 'm=1, n=2': (0.7205061947899576, 35566000), 'm=1, n=3': (0.7205061947899566, 77298200), 'm=1, n=4': (0.7205061947899436, 123279300), 'm=1, n=5': (0.7205061947898396, 162281200), 'm=2, n=1': (0.7461241928269338, 61941700), 'm=2, n=2': (0.7487382742592785, 119151000), 'm=2, n=3': (0.7491794005009862, 224218500), 'm=2, n=4': (0.7492810592233551, 337523200), 'm=2, n=5': (0.7493101045726759, 437942400), 'm=3, n=1': (0.750094329002307, 101656400), 'm=3, n=2': (0.7536233389359723, 203119200), 'm=3, n=3': (0.7543258733209144, 372078900), 'm=3, n=4': (0.7545150281573403, 546675800), 'm=3, n=5': (0.7545771545991714, 712261400), 'm=4, n=1': (0.7511399615752448, 148211200), 'm=4, n=2': (0.7550189559651415, 290019400), 'm=4, n=3': (0.7558538147201087, 525740100), 'm=4, n=4': (0.7560980916609719, 754398300), 'm=4, n=5': (0.7561850690701069, 1013667300), 'm=5, n=1': (0.7515075292697606, 195531200), 'm=5, n=2': (0.755539640616426, 370199500), 'm=5, n=3': (0.7564435298361336, 675432300), 'm=5, n=4': (0.7567210436674269, 972274200), 'm=5, n=5': (0.756824939133169, 1262492600)}

comp = 0.7575913566433811

color = ['#000000', '#11140B', '#23301C', '#2F4A2A', '#356636', '#368440', '#30A245', '#32C141', '#5EE032', '#97FC1A']

# Dirty data extraction
points = []
n_range = range(1, 5 + 1)
for i in n_range:
    points.append(data[f"m=1, n={i}"])
result_1 = [x[0] for x in points]
times_1 = [x[1] for x in points]
points.clear()
for i in n_range:
    points.append(data[f"m=2, n={i}"])
result_2 = [x[0] for x in points]
times_2 = [x[1] for x in points]
points.clear()
for i in n_range:
    points.append(data[f"m=3, n={i}"])
result_3 = [x[0] for x in points]
times_3 = [x[1] for x in points]
points.clear()
for i in n_range:
    points.append(data[f"m=4, n={i}"])
result_4 = [x[0] for x in points]
times_4 = [x[1] for x in points]
points.clear()
for i in n_range:
    points.append(data[f"m=5, n={i}"])
result_5 = [x[0] for x in points]
times_5 = [x[1] for x in points]
points.clear()

# Plotting
fig, ax = plt.subplots(facecolor='#000000')
plt.plot(n_range, np.asarray(result_1), label="m=1", c=color[3], lw=3)
plt.plot(n_range, np.asarray(result_2), label="m=2", c=color[4], ls= "--", lw=3)
plt.plot(n_range, np.asarray(result_3), label="m=3", c=color[5], ls = ":", lw=3)
plt.plot(n_range, np.asarray(result_4), label="m=4", c=color[6], ls = "-.", lw=3)
plt.plot(n_range, np.asarray(result_5), label="m=5", c=color[7], ls = (0, (3, 1, 1, 1, 1, 1)), lw=3)

ax.axhline(y=comp, color=color[-1], linestyle=':', label="$\psi_{1010} result$")
ax.xaxis.label.set_color("#5EE032")
ax.yaxis.label.set_color("#5EE032")
ax.tick_params(axis="x", colors="#5EE032")
ax.tick_params(axis="y", colors="#5EE032")
ax.set_facecolor("#000000")

for spine in ax.spines.keys():
    ax.spines[spine].set_color("#5EE032")

ax.grid(c="#5EE032", alpha=0.2)
plt.xlabel("n")
plt.ylabel("Result")
plt.legend(facecolor='#000000', edgecolor="#97FC1A", labelcolor='#5EE032')
plt.title("Scaling of the result with respect to n", color="#5EE032")
plt.show()