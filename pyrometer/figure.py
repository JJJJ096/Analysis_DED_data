import matplotlib.pyplot as plt


fig = plt.figure()
plt.subplots_adjust(hspace = 0.4, wspace = 0.3)
ax1 = plt.subplot(221)
ax1.set_xlabel("DOE variation")
ax1.set_ylabel("Average Temperature")
ax1.set_xlim([1,125])
ax1.set_ylim([1200,2000])

ax2 = plt.subplot(222)
ax2.set_xlabel("Energy Density[W/(mm/min)]")
ax2.set_ylabel("Average Temperature")
ax2.set_xlim([10,100])
ax2.set_ylim([1200,2000])

ax3 = plt.subplot(223)
ax3.set_xlabel("Temperature")
ax3.set_ylabel("Intensity")
ax3.set_xlim([1200,2000])
ax3.set_ylim([1,10000])

ax4 = plt.subplot(224)
ax4.set_xlabel("")
ax4.set_ylabel("")

plt.show()