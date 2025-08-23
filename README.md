# Johnaconda-ComfyUI-Custom-Nodes
Ok Theese are some Custom Nodes that i like to experiment with and use in ComfyUI.
Most of my work here is cooked up with the help of ChatGPT 5. i come up with neat ideas and we cook.

# How to install??
Oh this is easy! you just download the .py file or copy pasta the source into ComfyUi's custom nodes folder:
C:\ComfyUI\custom_nodes <- in here and BAM it should work, if it doesen't comfy probably have patched thier dependancy somehow that messed it up.

<img src="https://raw.githubusercontent.com/Johnaconda/Johnaconda-ComfyUI-Custom-Nodes/refs/heads/main/images/Howtoinstall.png?raw=true" alt="CombinerExample" title="Combiner preview usage">

# prompt_combiner_node.py
This is a Prompt combiner where it takes above string and second string and add them together.
it's a basic string combiner that allows me to add positives on top and leave em there while i edit the second string field "what to make"

<img src="https://raw.githubusercontent.com/Johnaconda/Johnaconda-ComfyUI-Custom-Nodes/refs/heads/main/images/Promptcombiner.png?raw=true" alt="CombinerExample" title="Combiner preview usage">

# The Ultra Branch
Ok i decided i wanted more "tune options" on the models without fucking em up too much so i made this!
its a sampler with two addon nodes, the sampler works on its own but with the help of the addon nodes,
we can bring out even more details from the same seed and model! i introduce ultra noise sweeps!
this node injects noise in the sampling process tricking the model there is even more details or atleast more gibberish!
fun to play around with even added some pre-defined modes when use-preset is true it overrides user defined settings
and also a special_CFG interpritor/injector that also will change the outcome of the image!

Added support for Scripts xy-plot and hires from https://github.com/jags111/efficiency-nodes-comfyui aswell.

<img src="https://raw.githubusercontent.com/Johnaconda/Johnaconda-ComfyUI-Custom-Nodes/refs/heads/main/images/comfyuishowcase.png?raw=true" alt="CombinerExample" title="Combiner preview usage">


# The SDXL Dual Model merger

this is a double node for merging models with a little more control, have fun with it.

<img src="https://raw.githubusercontent.com/Johnaconda/Johnaconda-ComfyUI-Custom-Nodes/refs/heads/main/images/merger2.png?raw=true" alt="merger" title="The merger">

<img src="https://raw.githubusercontent.com/Johnaconda/Johnaconda-ComfyUI-Custom-Nodes/refs/heads/main/images/Merger.png?raw=true" alt="merger workflow" title="merger workflow preview usage">
