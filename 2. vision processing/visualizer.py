import samna, samnagui
import time
from multiprocessing import Process
import signal
from functools import partial
import socket

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


def graceful_shutdown(interface, sig, frame):
    interface.graph.stop()
    print("\nClosing visualizer!")
    samna.device.close_device(interface.board)
    interface.visualizer_process.join()


def run_visualizer_process(
    sender_endpoint,
    receiver_endpoint,
    visualizer_id,
    window_height,
    window_width
):
    visualizer_process = Process(
        target=samnagui.runVisualizer,
        args=(
            window_width, 
            window_height, 
            receiver_endpoint, 
            sender_endpoint, 
            visualizer_id)
    )
    visualizer_process.start()
    return visualizer_process

class SamnaInterface:
    def __init__(self):
        self.graph = samna.graph.EventFilterGraph() 
        self.visualizer_process = None
        self.board = None

    def run_gui(self):
        samna_node = samna.init_samna(timeout_milliseconds=3000)
        sender_endpoint = samna_node.get_sender_endpoint()
        receiver_endpoint = samna_node.get_receiver_endpoint()

        # Get connected devices
        unopened_devices = samna.device.get_unopened_devices()

        # Make sure to get the correct device
        for device_info in unopened_devices:
            if device_info.device_type_name == "Speck2bTestboard":
                self.board = samna.device.open_device(device_info)

        # Check if board is initialized
        assert self.board

        # Configure the board to get sensor events
        config = samna.speck2b.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_sensor_enable = True
        self.board.get_model().apply_configuration(config)

        # Set up filter chain from board to visualizer
        ## Run samna gui in separate process
        visualizer_id = 3
        window_height = 0.75
        window_width = 0.75

        visualizer_process = run_visualizer_process(
            sender_endpoint=sender_endpoint,
            receiver_endpoint=receiver_endpoint,
            visualizer_id=visualizer_id,
            window_height=window_height,
            window_width=window_width
        )
        self.visualizer_process = visualizer_process

        ## Wait will opening remote node
        time.sleep(1)

        ## Get a free port
        port = get_free_port()

        ## Make visualizer
        samna.open_remote_node(visualizer_id, "visualizer")
        samna.visualizer.receiver.set_receiver_endpoint(f"tcp://0.0.0.0:{port}")
        samna.visualizer.receiver.add_destination(samna.visualizer.splitter.get_input_channel())

        ## Make streamer
        _, _, streamer = self.graph.sequential(
            [
                self.board.get_model_source_node(), 
                "Speck2bDvsToVizConverter",
                "VizEventStreamer"
            ]
        )
        # streamer = graph.get_node(streamer_id)
        streamer.set_streamer_endpoint(f"tcp://0.0.0.0:{port}")

        ## Make activity plot
        activity_plot_id = samna.visualizer.plots.add_activity_plot(
            128, 
            128, 
            "Dvs Visualization"
        )
        activity_plot = getattr(samna.visualizer, f"plot_{activity_plot_id}")
        activity_plot.set_layout(0, 0, 1, 1)

        samna.visualizer.splitter.add_destination(
            "dvs_event", 
            samna.visualizer.plots.get_plot_input(activity_plot_id)
        )

        # Start the graph
        self.graph.start()

        print("Built GUI")


# Define visualizer
visualizer = SamnaInterface()

# Define interrupt handler
signal.signal(
    signal.SIGINT, 
    partial(
        graceful_shutdown, 
        visualizer
    )
)


# Run visualizer
visualizer.run_gui()