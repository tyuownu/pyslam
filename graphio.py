#encoding=utf-8
import numpy as np
import textwrap
from future.utils import listvalues

from graph import Graph, VertexXYT, XYTConstraint
from mmath import MultiVariateGaussian



class GraphIOError(Exception):
    pass


# TODO: 添加关于3D部分.
def load_graph_g2o(filename):
    from collections import OrderedDict
    vertices = OrderedDict()
    edges = []

    with open(filename) as f:
        # 返回一个可遍历的索引序列，同时列出数据和数据下标.
        for linenum, line in enumerate(f):
            d = line.split()

            i = lambda idx: int(d[idx])
            f = lambda idx: float(d[idx])

            if d[0] == 'VERTEX_SE2':
                # ID
                id_ = i(1)
                # X, Y, theta
                xyt = np.array([ f(2), f(3), f(4) ])
                vertices[id_] = VertexXYT(xyt)

            elif d[0] == 'EDGE_SE2':
                # ID1, ID2
                ndx_out, ndx_in = i(1), i(2)
                v_out, v_in = vertices[ndx_out], vertices[ndx_in]

                # X, Y, theta
                xyt = np.array([ f(3), f( 4), f( 5) ])
                ## 信息矩阵.
                C = np.array([ [ f(6), f( 7), f( 8) ],
                               [ f(7), f( 9), f(10) ],
                               [ f(8), f(10), f(11) ] ])
                # linalg表示python numpy里面解线性代数的库.
                # inv表示求逆.
                g = MultiVariateGaussian(xyt, np.linalg.inv(C))
                edges.append( XYTConstraint(v_out, v_in, g) )

            else:
                msg = "Unknown edge or vertex type %s in line %d" % (d[0], linenum)
                raise GraphIOError(msg)

    # listvalues可以给vertices排序，按照id.
    return Graph(listvalues(vertices), edges)


def load_graph_toro(filename):
    from collections import OrderedDict
    vertices = OrderedDict()
    edges = []

    with open(filename) as f:
        for linenum, line in enumerate(f):
            d = line.split()

            i = lambda idx: int(d[idx])
            f = lambda idx: float(d[idx])

            if d[0] == 'VERTEX2':
                id_ = i(1)
                xyt = np.array([ f(2), f(3), f(4) ])
                vertices[id_] = VertexXYT(xyt)

            elif d[0] == 'EDGE2':
                ndx_out, ndx_in = i(1), i(2)
                v_out, v_in = vertices[ndx_out], vertices[ndx_in]

                xyt = np.array([ f( 3), f( 4), f( 5) ])
                P = np.array([ [ f( 6), f( 7), f(10) ],
                               [ f( 7), f( 8), f(11) ],
                               [ f(10), f(11), f( 9) ] ])
                g = MultiVariateGaussian(xyt, P)
                edges.append( XYTConstraint(v_out, v_in, g) )

            else:
                msg = "Unknown edge or vertex type %s in line %d" % (d[0], linenum)
                raise Exception(msg)

    return Graph(listvalues(vertices), edges)


def load_graph_april(filename):
    linenum = [0] # array because it is modified inside a nested function

    with open(filename) as f:
        def next_content_line():
            """ Ignore comments and empty lines """
            while True:
                line = f.readline()
                linenum[0] = linenum[0] + 1

                # Did we reach the end?
                if len(line) == 0: return None

                # skip empty lines and comments
                line = line.strip()
                if len(line) == 0: continue
                if line[0] == '#': continue

                return line

        def next_content_line_must_be(string):
            line = next_content_line()
            if line != string:
                raise GraphIOError('Expected %s on line %d. But found %s' % (
                                        string, linenum[0], line))

        vertices = []
        edges = []

        while True:
            line = next_content_line()
            if line is None: break

            if line == '"april.graph.GXYTNode"':
                next_content_line_must_be('{')
                next_content_line_must_be('vec 3')

                p = next_content_line().split()
                xyt = np.array([ float(p[0]), float(p[1]), float(p[2]) ])

                vertices.append(VertexXYT(xyt))

            if line == '"april.graph.GXYTEdge"':
                next_content_line_must_be('{')

                v_out = vertices[ int(next_content_line()) ]
                v_in = vertices[ int(next_content_line()) ]

                next_content_line_must_be('vec 3')
                p = next_content_line().split()
                xyt = np.array([ float(p[0]), float(p[1]), float(p[2]) ])

                next_content_line_must_be('vec -1')
                next_content_line_must_be('mat 3 3')
                p = next_content_line().split()
                a, b, c = float(p[0]), float(p[1]), float(p[2])
                p = next_content_line().split()
                d, e, f_ = float(p[0]), float(p[1]), float(p[2])
                p = next_content_line().split()
                g, h, i = float(p[0]), float(p[1]), float(p[2])

                C = np.array([ [ a, b, c ],
                               [ d, e, f_],
                               [ g, h, i ] ])

                P = np.linalg.inv(C)
                g = MultiVariateGaussian(xyt, P)
                edges.append( XYTConstraint(v_out, v_in, g) )

        return Graph(vertices, edges)



def load_graph(filename):
    if filename.endswith('.g2o'):
        return load_graph_g2o(filename)
    elif filename.endswith('.toro'):
        return load_graph_toro(filename)
    elif filename.endswith('.april'):
        return load_graph_april(filename)
    else:
        raise GraphIOError('Unsupported file input format "' +
            '%s". Valid extensions are ".g2o" ".toro" and ".april"' % filename)


_interactive_svg_doc = textwrap.dedent(
"""\
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>pySLAM -- SLAM graph viewer</title>
    <style>

    .overlay {
      fill: none;
      pointer-events: all;
    }

    polygon {
        fill: rgba(95, 158, 160, .8);
        stroke: navy;
        stroke-width: .1px;
    }

    line.odom {
        stroke: rgba(0, 0, 0, .25);
        stroke-width: .05px;
    }

    line.loop {
        stroke: rgba(192, 0, 0, .5);
        stroke-width: .1px;
    }

    body {
        overflow: hidden;
    }

    </style>
    <body>
    <script src="http://d3js.org/d3.v3.min.js"></script>
    <script>

    var width = window.innerWidth,
        height = window.innerHeight;

    var randomX = d3.random.normal(width / 2, 80),
        randomY = d3.random.normal(height / 2, 80);

    var vertices = %s;

    var edges_odom = %s;

    var edges_loop = %s;

    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height)
      .append("g")
        .attr("transform", "translate(" + width/2 + "," + height/2 + ")")
        .call(d3.behavior.zoom().scaleExtent([.1, 64]).on("zoom", zoom))
      .append("g");

    svg.append("rect")
        .attr("class", "overlay")
        .attr("transform", "translate(" + -width/2 + "," + -height/2 + ")")
        .attr("width", width)
        .attr("height", height);

    svg.selectAll("line.odom")
        .data(edges_odom)
      .enter().append("line")
        .classed("odom", true)
        .attr("x1", function(d) { return d[0]; })
        .attr("y1", function(d) { return d[1]; })
        .attr("x2", function(d) { return d[2]; })
        .attr("y2", function(d) { return d[3]; });

    svg.selectAll("line.loop")
        .data(edges_loop)
      .enter().append("line")
        .classed("loop", true)
        .attr("x1", function(d) { return d[0]; })
        .attr("y1", function(d) { return d[1]; })
        .attr("x2", function(d) { return d[2]; })
        .attr("y2", function(d) { return d[3]; });

    svg.selectAll("polygon")
        .data(vertices)
      .enter().append("polygon")
        .attr("points", "-1,1 -1,-1, 2,0")
        .attr("transform", function(d) { return "translate("+d[0]+","+d[1]+") rotate("+d[2]+")"; })
      .append("svg:title")
        .text(function(d, i) { return "" + i; });

    function zoom() {
      svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    </script>
""")


def render_graph_html(graph, filename=None):
    """
    Make a SVG rendering of `graph` and embed it in a .html file. The
    html file uses javascript to zoom and pan the SVG.

    Parameters:
    -----------
        `filename`: path of the created .html file
            If `filename` is `None`, return the generated html as a
            string.

    Returns:
    --------
        generated html content if the `filename` parameter is not
        specified.
    """
    _vertex_index_map = { v: i for i, v in enumerate(graph.vertices) }
    def ndx(v):
        return _vertex_index_map[v]

    xyt_constraints = [ e for e in graph.edges if isinstance(e, XYTConstraint) ]
    odom_constraints = [ e for e in xyt_constraints if abs(ndx(e._vx[0]) - ndx(e._vx[1])) == 1 ]
    loop_constraints = [ e for e in xyt_constraints if abs(ndx(e._vx[0]) - ndx(e._vx[1])) != 1 ]

    def vertex_xyt(v):
        return [ v.state[0]*10, v.state[1]*10, np.degrees(v.state[2]) ]

    def edge_coords(e):
        return [ e._vx[0].state[0]*10, e._vx[0].state[1]*10,
                  e._vx[1].state[0]*10, e._vx[1].state[1]*10 ]

    doc = _interactive_svg_doc % (
            [ vertex_xyt(v) for v in graph.vertices if isinstance(v, VertexXYT) ],
            [ edge_coords(e) for e in odom_constraints ],
            [ edge_coords(e) for e in loop_constraints ]
        )

    if filename is None:
        return doc
    else:
        with open(filename, 'w') as f:
            f.write(doc)
