import tkinter as tk
import torch
from torch import nn
import numpy as np


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        layer1 = self.relu(self.fc1(x))
        layer2 = self.relu(self.fc2(layer1))
        output = self.softmax(self.fc3(layer2))
        return layer1, layer2, output


model = NeuralNet()
state_dict = torch.load("./model/mnist_model.zip",
                        weights_only=False)
model.load_state_dict(state_dict)
model.eval()


class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Digit Recognition - Realtime Neural Network Visualizer")
        self.neuron_size = 16
        self.neuron_x_factor = self.neuron_size * 1.2
        self.font_size = 10
        self.canvas_color = "#222"
        self.root.configure(bg=self.canvas_color)
        self.visualizer_height = 700
        self.neuron_jump_factor = 3
        # self.root.attributes('-fullscreen', True)
        self.root.geometry("%dx%d" % (
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons/labels
        self.clear_button = tk.Button(
            root, text="Clear", command=self.clear_canvas, width=10)
        self.clear_button.grid(row=1, column=0)

        self.prediction_label = tk.Label(
            root, text="Prediction: ?", font=("Helvetica", 16))
        self.prediction_label.grid(row=2, column=0)

        # Neural network visualization canvas
        self.network_canvas = tk.Canvas(
            root, width=990, height=self.visualizer_height, bg=self.canvas_color, highlightthickness=0)
        self.network_canvas.grid(row=0, column=1, padx=10, pady=10, rowspan=3)

        self.pixel_size = 10
        self.grid = np.zeros((28, 28), dtype=np.float32)

    def draw(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x // self.pixel_size, y // self.pixel_size

        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            self.grid[grid_y, grid_x] = 1
            self.canvas.create_rectangle(
                grid_x * self.pixel_size,
                grid_y * self.pixel_size,
                (grid_x + 1) * self.pixel_size,
                (grid_y + 1) * self.pixel_size,
                fill="black",
            )
            self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = np.zeros((28, 28), dtype=np.float32)
        self.prediction_label.config(text="Prediction: ?")
        # self.network_canvas.delete("all")

    def predict_digit(self):
        input_tensor = torch.tensor(
            self.grid, dtype=torch.float32).unsqueeze(0).view(-1, 28 * 28)

        with torch.no_grad():
            layer1, layer2, output = model(input_tensor)
            predicted_digit = torch.argmax(output).item()

        self.prediction_label.config(text=f"Prediction: {predicted_digit}")
        self.visualize_network(layer1, layer2, output)

    def visualize_network(self, layer1, layer2, output):
        self.network_canvas.delete("all")

        def draw_layer(neurons, y_offset, x_start, layer_name):
            neuron_positions = []
            for i, activation in enumerate(neurons):
                color_intensity = int(activation * 255)
                color = f"#ff{color_intensity:02x}{color_intensity:02x}"
                neuron_x = x_start + i * self.neuron_x_factor
                self.network_canvas.create_oval(
                    neuron_x,
                    y_offset,
                    neuron_x + self.neuron_size,
                    y_offset + self.neuron_size,
                    fill=color,
                    outline="black"
                )
                neuron_positions.append((neuron_x + 10, y_offset + 10))
                self.network_canvas.create_text(
                    neuron_x + 10,
                    y_offset - 10,
                    text=f"{i}",
                    font=("Helvetica", 8),
                    anchor="s",
                    fill="white"
                )
            self.network_canvas.create_text(
                x_start + (len(neurons) * 20) / 2,
                y_offset + 50,
                text=layer_name,
                font=("Helvetica", self.font_size),
                fill="white"
            )
            return neuron_positions

        def draw_connections(from_neurons, to_neurons, weights):
            for i, from_pos in enumerate(from_neurons[::self.neuron_jump_factor]):
                for j, to_pos in enumerate(to_neurons[::self.neuron_jump_factor]):
                    weight = weights[j * self.neuron_jump_factor,
                                     i * self.neuron_jump_factor]
                    weight_intensity = int(abs(weight) * 255)
                    color = f"#{weight_intensity:02x}ff{weight_intensity:02x}" if weight > 0 else f"#ff{weight_intensity:02x}{weight_intensity:02x}"
                    self.network_canvas.create_line(
                        from_pos[0], from_pos[1], to_pos[0], to_pos[1], fill=color
                    )

        # Normalization
        layer1_activations = layer1[0].numpy() / np.max(layer1[0].numpy())
        layer2_activations = layer2[0].numpy() / np.max(layer2[0].numpy())
        output_activations = output[0].numpy() / np.max(output[0].numpy())

        # Draw layers -- reverse order
        layer1_positions = draw_layer(
            layer1_activations, self.visualizer_height * 0.8, 2, "Input Layer")
        layer2_positions = draw_layer(
            layer2_activations, self.visualizer_height * 0.5, 2, "Hidden Layer / Layer 2")
        output_positions = draw_layer(
            output_activations, self.visualizer_height * 0.2, (1000 * 0.5) - (len(output_activations) * 10), "Output Layer")

        # Draw connections/weights
        draw_connections(layer1_positions, layer2_positions,
                         model.fc2.weight.detach().numpy())
        draw_connections(layer2_positions, output_positions,
                         model.fc3.weight.detach().numpy())


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()
