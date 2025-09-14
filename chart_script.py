import plotly.graph_objects as go
import json

# Data from the provided JSON
data = [
    {"File": "architectural_fundamentals.txt", "Word_Count": 11247, "Category": "Fundamentals"},
    {"File": "cad_technical_specifications.txt", "Word_Count": 8842, "Category": "Technical"},
    {"File": "building_materials_database.txt", "Word_Count": 9156, "Category": "Materials"},
    {"File": "computational_design_algorithms.txt", "Word_Count": 12394, "Category": "Algorithms"},
    {"File": "text_to_cad_implementation_guide.txt", "Word_Count": 15738, "Category": "Implementation"},
    {"File": "documentation_summary.txt", "Word_Count": 377, "Category": "Summary"}
]

# Shortened file names for x-axis (15 character limit)
file_names = [
    "Arch Fund",
    "CAD Tech Spec", 
    "Materials DB",
    "Design Algo",
    "Implementation",
    "Summary"
]

word_counts = [item["Word_Count"] for item in data]

# Brand colors in order
colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C", "#B4413C"]

# Create bar chart
fig = go.Figure(data=[
    go.Bar(
        x=file_names,
        y=word_counts,
        marker_color=colors,
        text=[f"{count/1000:.1f}k" if count >= 1000 else str(count) for count in word_counts],
        textposition='outside'
    )
])

# Update layout
fig.update_layout(
    title="Text-to-CAD Architecture Documentation",
    xaxis_title="Doc Files",
    yaxis_title="Word Count",
    showlegend=False
)

# Update y-axis to show values in thousands with proper range
fig.update_yaxes(
    range=[0, 16000],
    tickformat=".0f",
    tickvals=[0, 4000, 8000, 12000, 16000],
    ticktext=["0", "4k", "8k", "12k", "16k"]
)

# Apply required trace update
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

fig.show()