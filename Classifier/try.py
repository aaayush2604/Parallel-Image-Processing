import matplotlib.pyplot as plt

# Data for metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [0.884, 0.8809523809523809, 0.888, 0.8844621513944223]

# Create the bar graph
plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])

# Add labels and title
plt.ylim(0, 1)  # Limit y-axis to [0, 1] for better visualization
plt.title('Performance Metrics', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Metrics', fontsize=14)

# Add value labels on bars
for i, value in enumerate(values):
    plt.text(i, value + 0.01, f"{value:.3f}", ha='center', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
