import argparse
import torch
import torch.nn as nn
import os
from typing import Union
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.backend.dynapcnn import io as dynapcnn_io
import samna, samnagui
import socket
from multiprocessing import Process
import time

class ANN(nn.Sequential):
    def __init__(self, n_classes=10):
        super().__init__(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=2, padding=1, bias=False),  # 16, 64, 64
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 16, 32, 32
            nn.Dropout2d(0.1),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False),  # 32, 16, 16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 32, 8, 8
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),  # 64, 4, 4
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 64, 2, 2
            nn.Flatten(),
            nn.Dropout2d(0.5),
            
            nn.Linear(32*4*4, n_classes, bias=False),
        )


def get_free_port():
    """Get a port that is not used by the OS at the moment.

    Returns:
        str: Port address
    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(('0.0.0.0', 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1]
    free_socket.close()
    return port

def start_samnagui_visualization_process(
        sender_endpoint, 
        receiver_endpoint, 
        visualizer_id, 
        window_height, 
        window_width
    ):
    """Start Samna GUI in another process
    
    Args:
        sender_endpoint str: 
            Sender endpoint tcp address and port
        receiver_endpoint str: 
            Receiver endpoint tcp address and port
        visualizer_id int: 
            ID of the visualizer (This is necessary for launching multiple visualizations)
        window_height float: 
            Height proportion of the window
        window_width float: 
            Width proportion of the window
    """
    visualizer_process = Process(
        target=samnagui.runVisualizer,
        args=(window_width, window_height, receiver_endpoint, sender_endpoint, visualizer_id)
    )
    visualizer_process.start()
    time.sleep(2) # This is necessary to make sure that the process is running before trying to connect to it.
    samna.open_remote_node(
        visualizer_id, 
        "samna_gui_node"
    )


def get_model(
    model_path: Union[os.PathLike, str]
):
    """_summary_

    Args:
        model_path (Union[os.PathLike, str]):
            Path to the model

    Returns:
        DynapcnnCompatibleNetwork:
            Discretized model that can be run on the Speck2b chip
    """
    ann = ANN(n_classes=7)
    sinabs_model = from_model(model=ann, add_spiking_output=True, input_shape=(2, 128, 128))
    sinabs_model = torch.load(model_path)
    dynapcnn_model = DynapcnnCompatibleNetwork(
        snn=sinabs_model.spiking_model,
        input_shape=(2, 128, 128),
        dvs_input=True,
        discretize=True
    )
    return dynapcnn_model

def readout_event_maker(feature):
    """Make a readout event and send it in a list.

    Args:
        feature int: The predicted feature

    Returns:
        List[samna.ui.Readout]: Samna UI Readout Type event with given feature
    """
    e = samna.ui.Readout()
    e.feature = feature
    return [e]

def find_max_in_dictionary(feature_dictionary):
    """Find the maximum value of all keys in the dictionary and return both 
    the key and value

    Args:
        feature_dictionary (Dict[int, int]): 
            Dictionary that contains (int, int) pairs of feature number 
            and number of recorded spikes

    Returns:
        Tuple(int, int): Feature number and number of recorded spikes
    """
    max_feature = 0
    max_n_spikes = 0
    for feature, n_spikes in feature_dictionary.items():
        if n_spikes > max_n_spikes:
            max_feature = feature
            max_n_spikes = n_spikes
    return max_feature, max_n_spikes
    

def readout_callback(spikes):
    """Readout callback to pass to samna

    Args:
        spikes samna.speck2b.Spike: 
            Recorded samna spikes from the readout

    Returns:
        List[samna.ui.Readout]: Samna UI Readout type events
    """
    default_retval = 6  # Id of the class 'other'
    threshold = 10  # Threshold to return anything else
    returned_features = {}

    for spike in spikes:
        if spike.feature in returned_features:
            returned_features[spike.feature] += 1
        else:
            returned_features[spike.feature] = 1

    max_feature, max_n_spikes = find_max_in_dictionary(returned_features)
    if max_n_spikes > threshold:  # If sufficient events have been received of the feature that was most active
        return readout_event_maker(max_feature)
    else:  #  Return other class
        return readout_event_maker(default_retval)

class SamnaInterface:
    def __init__(
        self,
        device
    ):
        self.device = device
        self.graph = samna.graph.EventFilterGraph()
        self.visualizer = None
        
        # Build the GUI
        self.build_gui()
    
    def build_gui(self):
        
        # Initialize samna (This is not necessary for device API, but still required for visualizer API)
        samna_node = samna.init_samna(timeout_milliseconds=3000)
        time.sleep(1)  # wait for TCP server to get set up
        # samna_node = samna
        receiver_endpoint = samna_node.get_receiver_endpoint()
        sender_endpoint = samna_node.get_sender_endpoint()
        
        # Parameters for Samna process
        visualizer_id = 3
        dvs_height, dvs_width = 128, 128
        dvs_layout = [0, 0, 0.5, 0.75]
        readout_layout = [0.5, 0, 1, 0.5]
        spike_count_layout = [0.5, 0.5, 1, 1]
        power_monitor_layout = [0, 0.75, 0.5, 1]
        window_height, window_width = .5625, .75  # Taken from modelzoo window width and height
        feature_count = 7
        feature_names = [
            "background",
            "clap",
            "michael jackson",
            "stay alive",
            "star",
            "wave",
            "other"
        ]
        spike_collection_interval = 500
        readout_images_path = "./readout_images/"
        
        
        # Start samna visualization process
        start_samnagui_visualization_process(
            sender_endpoint=sender_endpoint, 
            receiver_endpoint=receiver_endpoint, 
            visualizer_id=visualizer_id, 
            window_height=window_height, 
            window_width=window_width
        ) 
        samna_gui_node = getattr(samna, "samna_gui_node")
        
        visualizer_port = get_free_port()
        
        # Start power monitor
        power_monitor = self.device.get_power_monitor()
        power_monitor.start_auto_power_measurement(50)
        
        # Filter chain for visualizing DVS events
        _, _, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bDvsToVizConverter", "VizEventStreamer"])
        
        # Filter chain for visualizing spike counts
        _, spike_collection_node, spike_count_node, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bSpikeCollectionNode", "Speck2bSpikeCountNode", streamer])
        spike_collection_node.set_interval_milli_sec(spike_collection_interval)
        spike_count_node.set_feature_count(feature_count)
        
        # Filter chain for visualizing network output
        _, spike_collection_node, readout_filter_node, streamer = self.graph.sequential([self.device.get_model_source_node(), spike_collection_node, "Speck2bCustomFilterNode", streamer])
        readout_filter_node.set_filter_function(readout_callback)
        
        # Filter chain for visualizing power measurement
        _, measurement_to_viz, streamer = self.graph.sequential([power_monitor.get_source_node(), "MeasurementToVizConverter", streamer])
        
        # Make TCP connection between filter and the visualizer
        streamer_endpoint_ip = "tcp://0.0.0.0:" + str(visualizer_port)
        streamer.set_streamer_endpoint(streamer_endpoint_ip)
        
        samna_gui_node.receiver.set_receiver_endpoint(streamer_endpoint_ip)
        samna_gui_node.receiver.add_destination(samna_gui_node.splitter.get_input_channel())
        
        # Set up DVS Plot
        dvs_plot_id = samna_gui_node.plots.add_activity_plot(
            dvs_width,
            dvs_height,
            "DVS Plot"
        )
        dvs_plot = getattr(samna_gui_node, f"plot_{dvs_plot_id}")
        dvs_plot.set_layout(*dvs_layout)
        
        # Set up spike count plot
        spike_count_plot_id = samna_gui_node.plots.add_spike_count_plot(
            "Spike Count Plot",
            feature_count,
            feature_names
        )
        spike_count_plot = getattr(samna_gui_node, f"plot_{spike_count_plot_id}")
        spike_count_plot.set_layout(*spike_count_layout)
        spike_count_plot.set_show_x_span(25)
        spike_count_plot.set_label_interval(2.5)
        spike_count_plot.set_max_y_rate(1.2)
        spike_count_plot.set_show_point_circle(True)
        spike_count_plot.set_default_y_max(10)
        
        # Set readout plot
        ## Get images absolute paths
        images = []
        for image in os.listdir(readout_images_path):
            images.append(os.path.abspath(readout_images_path + image))
        images = sorted(images, key=lambda path: path.split('/')[-1])
        
        ## Set readout the plot
        readout_plot_id = samna_gui_node.plots.add_readout_plot(
            "Readout Plot",
            images
        )
        readout_plot = getattr(samna_gui_node, f"plot_{readout_plot_id}")
        readout_plot.set_layout(*readout_layout)
        
        # Set up power measurement plot
        power_measurement_plot_id = samna_gui_node.plots.add_power_measurement_plot(
            "Power consumption",
            3,  # Speck2b has 5 readouts
            ["io", "ram", "logic"]
        )
        power_measurement_plot = getattr(samna_gui_node, f"plot_{power_measurement_plot_id}")
        power_measurement_plot.set_layout(*power_monitor_layout)
        power_measurement_plot.set_show_x_span(10)
        power_measurement_plot.set_label_interval(2)
        power_measurement_plot.set_max_y_rate(1.5)
        power_measurement_plot.set_show_point_circle(False)
        power_measurement_plot.set_default_y_max(1)
        power_measurement_plot.set_y_label_name("power (mW)")
        # TODO: Set layout
        
        
        # Set splitters
        samna_gui_node.splitter.add_destination("dvs_event", samna_gui_node.plots.get_plot_input(dvs_plot_id))
        samna_gui_node.splitter.add_destination("spike_count", samna_gui_node.plots.get_plot_input(spike_count_plot_id))
        samna_gui_node.splitter.add_destination("readout", samna_gui_node.plots.get_plot_input(readout_plot_id))
        samna_gui_node.splitter.add_destination("measurement", samna_gui_node.plots.get_plot_input(power_measurement_plot_id))
        
        # Assign the values to class attributes for permanent access
        self.visualizer = samna_gui_node
        
        # Start the graph
        self.graph.start()
        
        # Prevent the operation from going out of scope
        while True:
            time.sleep(0.0001)

def main():
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Run GUI with the model path")
    parser.add_argument("model_path", help="Path to the model")
    device_id = "speck2b:0"
    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    
    config_modifier = {}
    
    # Load model from weights
    dynapcnn_model = get_model(model_path=model_path)
    dynapcnn_model.to(
        device_id,
        monitor_layers=["dvs", -1],
        # config_modifier=config_modifier
    )
    samna_device = dynapcnn_model.samna_device
    samna_interface = SamnaInterface(device=samna_device)
    


if __name__ == "__main__":
    main()