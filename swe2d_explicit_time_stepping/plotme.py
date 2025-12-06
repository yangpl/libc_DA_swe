import numpy as np
import matplotlib.pyplot as plt

nx, ny, nt = 101, 101, 500
dx, dy, dt = 10, 10, 0.1

filename = "zxy.bin"
dtype = np.float64

# open file once
with open(filename, "rb") as f:
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    data = np.fromfile(f, dtype=dtype, count=nx * ny)
    zxy = data.reshape((ny, nx))
    
    im = ax.imshow(zxy, extent=[0, nx*dx, 0, ny*dy], cmap='seismic', aspect='equal', origin='lower')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('SWE z(x,y)')
    plt.tight_layout()
    
    # loop through time frames
    for k in range(1, nt):
        f.seek(k * nx * ny * np.dtype(dtype).itemsize)
        data = np.fromfile(f, dtype=dtype, count=nx * ny)    #read nx*ny*dtype data each time
        if data.size < nx * ny:
            break  # stop if file ended early
        
        zxy = data.reshape((ny, nx))
        im.set_data(zxy)
        ax.set_title(f'SWE z(x,y) — Frame {k}/{nt}')
        plt.pause(0.05)  # short delay to update the plot

plt.ioff()
plt.show()

