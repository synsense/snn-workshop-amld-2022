import argparse
import torch
import torch.nn as nn
import os
import numpy as np
from typing import Union, Tuple, List
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn import io as dynapcnn_io
import samna, samnagui
import socket
import yaml
from multiprocessing import Process
from functools import partial
import time
import signal

class ANN(nn.Sequential):
    def __init__(self, n_classes=10):
        super().__init__(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=2, padding=1, bias=False),  # 16, 64, 64
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


class HotpixelFilter:
    """
        The original description of the algorithm by Alejandro Linares-Barranco et al. in the following paper.
        - https://www.zora.uzh.ch/id/eprint/184205/1/08836544.pdf
    """
    def __init__(
        self,
        device,
        event_count_threshold: int,
        transition_state_threshold: int,
        dvs_resolution: Tuple[int, int]
    ):
        self.device = device
        self.event_count_threshold = event_count_threshold
        self.transition_state_threshold = transition_state_threshold
        self.dvs_map = np.zeros(shape=dvs_resolution, dtype=int)
        
    def assign_events(
        self,
        events: List[samna.speck2b.event.Spike]
    ):
        for event in events:
            self.dvs_map[event.y, event.x] += 1
    
    def find_hotpixels(self):
        return np.where(self.dvs_map > self.event_count_threshold)
    
    def kill_hotpixels(self):
        source = samna.BasicSourceNode_speck2b_event_input_event()
        source.add_destination(self.device.get_model().get_sink_node().get_input_node())
        hotpixels = self.find_hotpixels()
        kill_events = []
        for y, x in zip(hotpixels[0], hotpixels[1]):
            e = samna.speck2b.event.KillSensorPixel()
            e.y=y
            e.x=x
            kill_events.append(e)
        print(f"Number of hotpixels killed: {len(kill_events)}")
        source.write(kill_events)
        

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
    time.sleep(4) # This is necessary to make sure that the process is running before trying to connect to it.
    samna.open_remote_node(
        visualizer_id, 
        "samna_gui_node"
    )


def get_model(
    model_path: Union[os.PathLike, str]
):
    """From the path to the model get a dynapcnn network

    Args:
        model_path (Union[os.PathLike, str]):
            Path to the model

    Returns:
        DynapcnnCompatibleNetwork:
            Discretized model that can be run on the Speck2b chip
    """
    ann = ANN(n_classes=7)
    sinabs_model = from_model(model=ann, add_spiking_output=True, input_shape=(1, 128, 128))
    sinabs_model = torch.load(model_path)
    dynapcnn_model = DynapcnnNetwork(
        snn=sinabs_model.spiking_model,
        input_shape=(1, 128, 128),
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

class SamnaInterface:
    def __init__(
        self,
        device
    ):
        self.device = device
        self.graph = samna.graph.EventFilterGraph()
        self.visualizer = None

        # Define these within the scope of being able to access configuration within
        self.configuration = None
        
        # Define these within the scope of the class for graceful shutdown
        self.readout_node = None
        self.power_monitor = None
    
        self.hotpixel_filter = HotpixelFilter(
            device=device, 
            transition_state_threshold=20000,
            event_count_threshold=20, 
            dvs_resolution=(128, 128)
        )
    
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
        dvs_layout = [0, 0, 0.5, 0.6]
        readout_layout = [0.5, 0, 1, 0.5]
        spike_count_layout = [0.5, 0.5, 1, 1]
        power_monitor_layout = [0, 0.6, 0.5, 1]
        window_height, window_width = .5625, .75  # Taken from modelzoo window width and height
        
        # Parameters loaded from configuration
        feature_count = self.configuration["feature_count"]
        feature_names = self.configuration["feature_names"]
        spike_collection_interval = self.configuration['spike_collection_interval']
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
        
        # Define within the scope of the class to close it.
        self.power_monitor = power_monitor
        
        # Filter chain for visualizing DVS events
        _, _, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bDvsToVizConverter", "VizEventStreamer"])
        
        # Filter chain for visualizing spike counts
        _, spike_collection_node, spike_count_node, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bSpikeCollectionNode", "Speck2bSpikeCountNode", streamer])
        spike_collection_node.set_interval_milli_sec(spike_collection_interval)
        spike_count_node.set_feature_count(feature_count)
        
        # Filter chain for visualizing network output
        _, spike_collection_node, readout_filter_node, streamer = self.graph.sequential([self.device.get_model_source_node(), spike_collection_node, "Speck2bCustomFilterNode", streamer])
        readout_filter_node.set_filter_function(self.readout_callback)
        self.readout_node = readout_filter_node # Assign to class to be able to close this for graceful shutdown!
        
        samna_camera_buffer = samna.BufferSinkNode_speck2b_event_output_event()
        _, event_type_filter, _ = self.graph.sequential([self.device.get_model_source_node(), "Speck2bOutputEventTypeFilter", samna_camera_buffer])
        event_type_filter.set_desired_type("speck2b::event::DvsEvent")  # 9 = events received from sensor

        
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
        
        
        # Set splitters
        samna_gui_node.splitter.add_destination("dvs_event", samna_gui_node.plots.get_plot_input(dvs_plot_id))
        samna_gui_node.splitter.add_destination("spike_count", samna_gui_node.plots.get_plot_input(spike_count_plot_id))
        samna_gui_node.splitter.add_destination("readout", samna_gui_node.plots.get_plot_input(readout_plot_id))
        samna_gui_node.splitter.add_destination("measurement", samna_gui_node.plots.get_plot_input(power_measurement_plot_id))
        
        # Assign the values to class attributes for permanent access
        self.visualizer = samna_gui_node
        
        # Start the graph
        self.graph.start()
        
        # Check and activate the hotpixel filter
        hotpixel_filter_activated_flag = False
        camera_events_received = []
        while True:
            camera_events_received.extend(samna_camera_buffer.get_events())
            if len(camera_events_received) > self.hotpixel_filter.transition_state_threshold and not hotpixel_filter_activated_flag:
                print(f"Hotpixel filter activated! at event: {len(camera_events_received)}")
                self.hotpixel_filter.assign_events(camera_events_received)
                self.hotpixel_filter.kill_hotpixels()
                hotpixel_filter_activated_flag = True
            elif hotpixel_filter_activated_flag:  # Deallocate received events as they are no longer useful.
                camera_events_received = []
            time.sleep(0.0001)
    
    def load_config(
        self, 
        configuration_path: Union[os.PathLike, str]
    ):
        """Load yaml configuration from path
        
        Args:
            configuration_path Union[os.PathLike, str]:
                Path to the configuration object
        """
        with open(configuration_path, "r") as f:
            self.configuration = yaml.safe_load(f)
    
    def readout_callback(
        self, 
        spikes
    ):
        """Readout callback to pass to samna

        Args:
            spikes samna.speck2b.Spike: 
                Recorded samna spikes from the readout

        Returns:
            List[samna.ui.Readout]: Samna UI Readout type events
        """
        default_retval = self.configuration['readout_callback']['default_readout']
        threshold = self.configuration['readout_callback']['threshold']
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


def graceful_shutdown(interface, sig, frame):
    interface.readout_node.stop()
    interface.graph.stop()
    interface.power_monitor.stop_auto_power_measurement()
    print('\n Shutting down interface!')
    exit(0)

def main():
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Run GUI with the model path")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("configuration_path", help="Path to the configuration file")
    device_id = "speck2b:0"
    
    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    configuration_path = args.configuration_path
    
    # Load model from weights
    dynapcnn_model = get_model(model_path=model_path)
    dynapcnn_model.to(
        device_id,
        monitor_layers=["dvs", -1],
        # config_modifier=config_modifier
    )
    samna_device = dynapcnn_model.samna_device
    samna_interface = SamnaInterface(device=samna_device)

    # Load configuration
    samna_interface.load_config(configuration_path)
    
    # Define interrupt handler
    signal.signal(signal.SIGINT, partial(graceful_shutdown, samna_interface))
    
    # Build GUI
    samna_interface.build_gui()
    

if __name__ == "__main__":
    main()
