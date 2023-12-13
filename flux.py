






# aa 

import graphviz

# Create a new Graphviz graph
dot = graphviz.Digraph(format='svg')

# Add nodes and edges to the graph
dot.node('Start', shape='ellipse')
dot.node('6!', label='6! (720)')
dot.node('5!', label='5! (120)')
dot.node('4!', label='4! (24)')
dot.node('3!', label='3! (6)')
dot.node('2!', label='2! (2)')
dot.node('1!', label='1! (1)')
dot.node('Sum', label='Sum = 873')
dot.node('Multiply', label='× 20')
dot.node('Result', label='Result = 17460', shape='ellipse')
dot.node('End', shape='ellipse')

dot.edges(['Start->6!', '6!->5!', '5!->4!', '4!->3!', '3!->2!', '2!->1!', '1!->Sum', 'Sum->Multiply', 'Multiply->Result', 'Result->End'])

# Save the graph as an SVG file
dot.render('flowchart', view=True)
























