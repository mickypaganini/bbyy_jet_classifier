import matplotlib.pyplot as plt

def atlaslabel(ax, fontsize=20):
    # -- Add ATLAS text
    plt.text(0.1,0.9, 'ATLAS', va='bottom', ha='left', color='black', size=fontsize, 
             fontname = 'sans-serif', weight = 'bold', style = 'oblique',transform=ax.transAxes)
    plt.text(0.23, 0.9, 'Work In Progress', va='bottom', ha='left', color='black', size=fontsize,
            fontname = 'sans-serif', transform=ax.transAxes)
    plt.text(0.1, 0.83, 
             r'$\sqrt{s} = 13 TeV :\ \ \int Ldt = 1.04\ fb^{-1}$', # change number according to actual data used!
             va='bottom', ha='left', color='black', size=fontsize, fontname = 'sans-serif', transform=ax.transAxes)