"""Additional functions for the spatial separation demo."""

import numpy as np

import mne
import pyvista as pv

###################################################################################################
###################################################################################################

def load_helper(h5_file):
    """Helper function for loading data of interest from the h5 file."""

    # Extract channel names and electrode positions
    ch_names = get_channel_names(h5_file)
    electrodes = h5_file["sa"]["locs_3D"][:].T

    # Remove lower electrodes for aesthetical reasons
    idx_select = electrodes[:, 2] > -40
    electrodes = electrodes[idx_select, :]

    # Get the leadfield
    leadfield = h5_file["sa"]["cortex75K"]["V_fem_normal"][:]
    leadfield = leadfield[:, idx_select]
    ch_names = ch_names[idx_select]

    return ch_names, electrodes, leadfield


def get_pos(h5_file, field):
    """Get positions from h5 file for specified field.

    Returns
    -------
    pos : array, xyz-coordinates of nodes in mesh.
    """

    pos = h5_file["sa"][field]["vc"][:].T
    tri = h5_file["sa"][field]["tri"][:].T - 1

    return pos, tri


def get_channel_names(h5_file):
    """Flatten electrode names into list.

    Parameters
    ----------
    h5_file : H5 file object from which electrode names should be extracted.
    """

    ch_ref = [ch[0] for ch in h5_file["sa"]["clab_electrodes"][:]]
    ch_names = []
    for ch in ch_ref:
        ch_name = "".join([chr(ind[0]) for ind in h5_file[ch][:]])
        ch_names.append(ch_name)

    ch_names = np.array(ch_names)

    return ch_names


def get_channel_index(ch_names, channel):
    """Get index of requested channel."""

    return np.where(np.array(ch_names) == channel)


def get_leadfield_coefs(h5_file, ch1, ch2, selected_channel):
    """Get the leadfield coefficients for channels of interest."""

    # Grab data of interest from the h5 file
    ch_names, electrodes, leadfield = load_helper(h5_file)
    pos_brain, _ = get_pos(h5_file, "cortex75K")

    # Find dipole close to channel location: PP05 and then shift around a bit
    idx_source1 = get_dipole_location(channel=ch1, electrodes=electrodes,
                                      ch_names=ch_names, offset=[15, 0, 0], pos=pos_brain)
    idx_source2 = get_dipole_location(channel=ch2, electrodes=electrodes,
                                      ch_names=ch_names, offset=[10, 0, 0], pos=pos_brain)

    # Get the lead field weights a_i for the two sources for the chosen channel
    #   channel activity x_i is a linear mixture of the two sources s_1 and s_2
    #   x_i = a_1i * s_1 + a_2i * s_2
    leadfield_coef = leadfield[[idx_source1, idx_source2],
                               get_channel_index(ch_names, selected_channel)]

    return leadfield_coef, idx_source1, idx_source2


def get_dipole_location(channel, electrodes, ch_names, pos, offset=(0, 0, 0)):
    """Returns the index of the the mesh node closest to a chosen electrode position with
    a potential offset. Convenience function for quickly locating dipoles.

    Parameters
    ----------
    channel : str, name of channel the source should be close to.
    electrodes : array (n_channels x 6) coordinates of electrodes.
    ch_names : list (n_channels), list of channel names.
    offset : list (3), xyz-coordinates of offset with respect to channel.
    pos : coordinates of gray matter mesh.

    Returns
    -------
        idx_source : index of node in gray matter mesh
    """

    idx_chan = np.where(np.array(ch_names) == channel)
    xyz_chan = electrodes[idx_chan[0], :3]
    xyz_chan += np.array(offset)[np.newaxis]
    idx_source = np.argmin(np.sum((pos - xyz_chan) ** 2, axis=1))

    return idx_source

###################################################################################################
###################################################################################################

def make_mne(epochs, fs):
    """Helper function to make an MNE object."""

    ch_signals = ["signal%i" % ind for ind in range(len(epochs))]
    info = mne.create_info(ch_signals, fs, ch_types="eeg")
    raw = mne.io.RawArray(epochs, info, verbose=False)

    return raw


def make_mne_topo(ch_names, electrodes, fs, leadfield, idx_source1, idx_source2):
    """Make an MNE object with leadfield topographies."""

    info = mne.create_info(list(ch_names), fs, "eeg")

    # Set leadfield as data & add to MNE object
    data = leadfield[[idx_source1, idx_source2], :].T
    raw = mne.io.RawArray(data, info, verbose=False)

    ch_pos = dict(zip(ch_names, electrodes[:, :3]))
    montage = mne.channels.make_dig_montage(ch_pos, coord_frame="head")
    raw.set_montage(montage);

    return raw


def plot_topography(raw, ind, selected_channel, colors):
    """Plot a leadfield topography from an MNE object."""

    # Mark selected channel in the topoplot
    idx_chan = np.where(np.array(raw.ch_names) == selected_channel)
    mask = np.zeros((len(raw.ch_names),), dtype="bool")
    mask[idx_chan] = True

    mask_params = dict(marker="o", markerfacecolor="w",
                       markeredgecolor=colors[2],
                       linewidth=0, markersize=5)

    mne.viz.plot_topomap(raw.get_data()[:, 0], raw.info,
                         mask=mask, mask_params=mask_params);

###################################################################################################
###################################################################################################

def plot_mesh(pos, tri, plotter, color, opacity=1):
    """Visualizes a given mesh from NYhead with pyvista.

    Parameters
    ----------
    plotter : pyvista plotter
    color : str, color of mesh.
    opacity : float, regulate opacity of mesh.
    """

    # structure for triangular faces as required by pyvista
    faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
    faces = faces.astype("int")

    cloud = pv.PolyData(pos, faces)
    actor = plotter.add_mesh(cloud, color=color, opacity=opacity)


def plot_electrodes(electrodes, plotter, color="w"):
    """Plot electrodes as small cyclinders into a specific pyvista plotter.

    Parameters
    ----------
    electrodes : array, (n_electrodes x 6) electrode coordinates and normals.
    plotter : pyvista plotter
    color : str, color of electrodes.
    """

    for ind in range(len(electrodes)):
        cylinder = pv.Cylinder(center=electrodes[ind, :3], direction=electrodes[ind, 3:],
                               radius=3.5, height=2.0)
        plotter.add_mesh(cylinder, color=color)


def plot_head(h5_file, ch1, ch2, selected_channel, save_out=False):
    """Plots the head reconstruction with highlighted channels and sources."""

    # Plot the 3d brain, with head and cortex meshes
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 1200))
    pos_head, tri_head = get_pos(h5_file, "head")
    plot_mesh(pos_head, pos_head, plotter=plotter, color="orange", opacity=0.3)
    pos_brain, tri_brain = get_pos(h5_file, "cortex75K")
    plot_mesh(pos_brain, tri_brain, plotter=plotter, color="#AAAAAA", opacity=1)

    # Get the leadfield coefficients
    leadfield_coef, idx_source1, idx_source2 = get_leadfield_coefs(\
        h5_file, ch1, ch2, selected_channel)

    # Plot the 2 dipole locations
    plotter.add_mesh(pos_brain[[idx_source1], :],
                     render_points_as_spheres=True, point_size=40, color="r")
    plotter.add_mesh(pos_brain[[idx_source2], :],
                     render_points_as_spheres=True, point_size=40, color="b")

    # Plot electrodes, highlighting selected electrode in green
    ch_names, electrodes, leadfield = load_helper(h5_file)
    plot_electrodes(electrodes, plotter, color="w")
    plot_electrodes(electrodes[get_channel_index(ch_names, selected_channel)],
                    plotter, color="green")

    # Set the camera position for 3d visualization
    r = 650
    angle = 3.6
    cpos = [(r * np.sin(angle), r * np.cos(angle), 50), (0, 0, 0), (0, 0, 1)]

    if save_out:
        plotter.set_background(None)
        plotter.show(cpos=cpos, interactive_update=False, screenshot='mesh_nyhead.png')
