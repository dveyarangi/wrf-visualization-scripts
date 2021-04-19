import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
def LetterChanges(x):
    # code goes here
    import string
    xnew = x 
    xnew = list(xnew)

    for l in range(len(x)):
        
        if x[l] == " ":
           xnew[l]="_"
    xnew = "".join(xnew) # Change the list back to string, by using 'join' method of strings.    
    return xnew

def plot_profile( line_profiles, scatter_profiles ,outdir, xlim, title="", prefix=""):
    if line_profiles is not None:
        for label in sorted(iter(line_profiles.values)):

            profile = line_profiles.values[label]

            if line_profiles.is_angular:
                updated_profile = []
                for theta in profile:
                    while theta < 360:
                        theta += 360
                    while theta > 360:
                        theta -= 360
                    updated_profile.append(theta)
                    profile = updated_profile

            plt.plot( profile, line_profiles.heights, label=label)
    if scatter_profiles is not None:
        for label in sorted(iter(scatter_profiles.values)):
            profile = scatter_profiles.values[label]
            if scatter_profiles.is_angular:
                updated_profile = []
                for theta in profile:
                    updated_profile.append(theta % 360)
                profile = updated_profile
            plt.scatter( profile, scatter_profiles.heights, label=label)

    plt.title(title)
    plt.legend(loc='best')
    plt.gca().xaxis.grid(True, linestyle='--')
    plt.gca().yaxis.grid(True, linestyle='--')

    if not xlim is None:
        (xmin, xmax) = xlim
        ymin = line_profiles.heights[0]
        ymax = line_profiles.heights[len(line_profiles.heights)-1]
        plt.axis([xmin, xmax, ymin, ymax])


    # padding outfile name
    label1=LetterChanges(label)
    if outdir is not None:
        plt.savefig(f'{outdir}/{prefix}.png')
    #plt.show()
    plt.close()

def plot_time_series( line_series, outdir, ylim, title="", prefix=""):
    if line_series is not None:
        for label in iter(line_series.values):

            series = line_series.values[label]
            datetimes = []
            xticks = []
            xlabels = []
            for idx, millis in enumerate(line_series.xs):

                dt_label = dt.datetime.utcfromtimestamp(millis/1000)
                datetimes.append(dt_label)
                if dt_label.hour % 6 == 0 and dt_label.minute == 0:
                    xticks.append(dt_label)
                    xlabels.append(dt_label.strftime("%d-%m %HZ"))


            dates = matplotlib.dates.date2num(datetimes)
            #if line_series.is_angular:

            #    updated_profile = []
            #    for theta in series:
            #        updated_profile.append(theta % 360)

            #    series = updated_profile = np.unwrap(updated_profile)
                #series = updated_profile

            plt.xticks(xticks, xlabels, rotation=30)
            plt.gca().xaxis.grid(True,linestyle='--')
            plt.gca().yaxis.grid(True, linestyle='--')
            plt.plot( datetimes, series, label=label)

    plt.title(title)
    plt.legend(loc='best')

    if not ylim is None:
        (ymin, ymax) = ylim
        xmin = line_series.xs[0]
        xmax = line_series.xs[len(line_series.xs)-1]
        plt.ylim([ymin, ymax])


    # padding outfile name
    label1=LetterChanges(label)
    if outdir is not None:
        plt.savefig(f'{outdir}/{prefix}.png', bbox_inches='tight')
    plt.close()

rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

def plot_bars( bar_series,outdir, ylim=None, title="", prefix="", side_legend=None):
    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab20").colors
    if bar_series is not None:
        series_num = len(bar_series.values)+1
        offset = series_num/2.
        xlabels = [date for date in bar_series.xs]
        for idx, label in enumerate(iter(bar_series.values)):

            series = bar_series.values[label]
            xticks = np.arange(len(bar_series.xs)-1)

            ax.bar(xticks+(idx+1)/series_num, series, width=1/series_num, label=label,
                   color=colors[idx*3%len(colors)])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=30)

            if not ylim is None:
                (ymin, ymax) = ylim
                ax.set_ylim([ymin, ymax])

    plt.gca().xaxis.grid(True, linestyle='--')
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.title(title)

    if side_legend is None:
        plt.legend(loc='best')
    else:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    # padding outfile name
    label1 = LetterChanges(label)
    if outdir is not None:
        plt.savefig(f'{outdir}/{prefix}.png', bbox_inches='tight')
    plt.close()

def plot_line_and_errors( series, error_series, xticks, xlabels, outdir, color=None,ylim=None, title="", prefix="", side_legend=None):
    fig, ax = plt.subplots()

    #xticks = np.arange(len(series.xs)-1)
    ax.errorbar(xticks, series, error_series, linewidth=2, ecolor='black', capsize=3)
   #ax.bar(xlabels, series, yerr=error_series, alpha=0.5, color=color, ecolor='black', capsize=3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    if not ylim is None:
        (ymin, ymax) = ylim
        ax.set_ylim([ymin, ymax])

    plt.gca().xaxis.grid(True, linestyle='--')
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.title(title)

    #if side_legend is None:
    #    plt.legend(loc='best')
    #else:
    #    plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    # padding outfile name
    if outdir is not None:
        plt.savefig(f'{outdir}/{prefix}.png', bbox_inches='tight')
    plt.close()

def plot_radial( line_profiles, scatter_profiles ,outdir, xlim, title="", prefix=""):
    fig = plt.figure()
    # Set the axes as polar
    ax = fig.add_subplot(111, polar=True)

    if line_profiles is not None:
        for label in sorted(iter(line_profiles.values)):
            profile = line_profiles.values[label]
            rad_profile = []
            for theta in profile:
                rad_profile.append(theta * 2 * np.pi / 360)
            plt.plot( rad_profile, line_profiles.heights, label=label)
    if scatter_profiles is not None:
        for label in sorted(iter(scatter_profiles.values)):
            profile = scatter_profiles.values[label]
            rad_profile = []
            for theta in profile:
                rad_profile.append(theta * 2 * np.pi / 360)
            plt.scatter( rad_profile, scatter_profiles.heights, label=label)

    plt.title(title)
    plt.legend(loc='best')


    # padding outfile name
    if outdir is not None:
        plt.savefig(outdir+prefix+'.png')
   # plt.show()
    plt.close()