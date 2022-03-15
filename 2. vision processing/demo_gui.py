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
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(('0.0.0.0', 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1]
    free_socket.close()
    return port

def start_samnagui_visualization_process(sender_endpoint, receiver_endpoint, visualizer_id, window_height, window_width):
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


def get_model(model_path: Union[os.PathLike, str]):
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

def readout_function(events):
    def return_result(feature):
        e = samna.graph.nodes.Readout()
        e.feature = feature
        return [e]
    
    raise NotImplementedError()

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
        samna_node_id = 1
        interpreter_id = 2
        visualizer_id = 3
        dvs_height, dvs_width = 128, 128
        dvs_layout = [0, 0, 0.5, 1]
        readout_layout = [0.5, 0, 1, 0.5]
        spike_count_layout = [0.5, 0.5, 1, 1]
        window_height, window_width = .5625, .75  # Taken from modelzoo window width and height
        feature_count = 7
        feature_names = [
            "background",
            "clap",
            "michael_jackson",
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
        
        # Filter chain for visualizing DVS events
        _, _, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bDvsToVizConverter", "VizEventStreamer"])
        
        # Filter chain for visualizing spike counts
        _, spike_collection_node, spike_count_node, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bSpikeCollectionNode", "Speck2bSpikeCountNode", streamer])
        spike_collection_node.set_interval_milli_sec(spike_collection_interval)
        spike_count_node.set_feature_count(feature_count)
        
        # Filter chain for visualizing network output
        _, readout_filter_node, streamer = self.graph.sequential([self.device.get_model_source_node(), "Speck2bCustomFilterNode", streamer])
        readout_filter_node.set_filter_function(...)
        
        
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
        
        ## Set the plot
        readout_plot_id = samna_gui_node.plots.add_readout_plot(
            "Readout Plot",
            images
        )
        readout_plot = getattr(samna_gui_node, f"plot_{readout_plot_id}")
        readout_plot.set_layout(*readout_layout)
        
        
        # Set splitters
        samna_gui_node.splitter.add_destination("dvs_event", samna_gui_node.plots.get_plot_input(dvs_plot_id))
        samna_gui_node.splitter.add_destination("spike_count", samna_gui_node.plots.get_plot_input(spike_count_plot_id))
        samna_gui_node.splitter.add_destination("readout", samna_gui_node.plots.get_plot_input(readout_plot_id))
        
        # Assign the values to class attributes for permanent access
        self.visualizer = samna_gui_node
        self.graph.start()
        
        while True:
            time.sleep(0.01)
        
        
        
    def run_model(
        self,
        model_path: Union[os.PathLike, str]
    ):
        pass

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