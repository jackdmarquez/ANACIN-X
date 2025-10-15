from functools import wraps
from time import time
import json
import multiprocessing as mp
import psutil
import igraph
import numpy as np

from typing import Optional

# Silence warnings from igraph
import warnings
warnings.filterwarnings("ignore", module="igraph")

# A simple decorator for timing function calls
def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):                                                  
        start = time()                                                             
        result = f(*args, **kwargs)                                                
        end = time()                                                               
        print("{} - Elapsed time: {}".format(f, end-start))
        return result                                                              
    return wrapper


def read_run_params( run_params_path ):
    with open(run_params_path, "r") as infile:
        return json.load(infile)

# A function to read in a single graph file via igraph
#@timer
COLLECTIVE_MPI_CALL_TO_EVENT_TYPE = {
    "BARRIER"   : "barrier",
    "IBARRIER"  : "ibarrier",
    "WAIT"      : "wait",
    "WAITALL"   : "waitall",
    "WAITANY"   : "waitany",
    "WAITSOME"  : "waitsome",
}

EVENT_TYPES_TO_PRESERVE = {"SEND", "RECV"}

MPI_CALL_ATTRIBUTE_CANDIDATES = (
    "mpi_call",
    "function",
    "mpi_function",
    "mpi_call_name",
    "operation",
    "call",
)


def _normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip()
    if len(normalized) == 0:
        return None
    normalized = normalized.split("(")[0]
    normalized = normalized.split("::")[-1]
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "_")
    normalized = normalized.upper()
    for prefix in ("PMPI_", "MPI_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def _extract_mpi_call_name(vertex) -> Optional[str]:
    attrs = vertex.attributes()
    for attr_name in MPI_CALL_ATTRIBUTE_CANDIDATES:
        value = attrs.get(attr_name)
        if isinstance(value, str) and len(value.strip()) > 0:
            return value
    return None


def normalize_event_types(graph):
    n_vertices = graph.vcount()
    if n_vertices == 0:
        if "event_type" not in graph.vs.attributes():
            graph.vs["event_type"] = []
        graph.vs["collective_type"] = []
        graph.vs["is_collective"] = []
        return graph

    vertex_attr_names = graph.vs.attributes()
    if "event_type" in vertex_attr_names:
        event_types = list(graph.vs["event_type"])
    else:
        event_types = [None] * n_vertices

    collective_types = [None] * n_vertices
    is_collective = [False] * n_vertices

    for idx, vertex in enumerate(graph.vs):
        current_type = event_types[idx]
        normalized_current_type = _normalize_label(current_type)

        mpi_call_name = _extract_mpi_call_name(vertex)
        normalized_call = _normalize_label(mpi_call_name)

        new_type = None
        if normalized_call in COLLECTIVE_MPI_CALL_TO_EVENT_TYPE:
            new_type = COLLECTIVE_MPI_CALL_TO_EVENT_TYPE[ normalized_call ]
        elif normalized_current_type in COLLECTIVE_MPI_CALL_TO_EVENT_TYPE:
            new_type = COLLECTIVE_MPI_CALL_TO_EVENT_TYPE[ normalized_current_type ]

        if new_type is not None:
            if normalized_current_type not in EVENT_TYPES_TO_PRESERVE:
                event_types[idx] = new_type
            is_collective[idx] = True
            collective_types[idx] = new_type

    graph.vs["event_type"] = event_types
    graph.vs["collective_type"] = collective_types
    graph.vs["is_collective"] = is_collective
    return graph


def read_graph( graph_path ):
    graph = igraph.read( graph_path )
    normalize_event_types( graph )
    return graph

def read_graphs_serial( graph_paths ):
    return [ read_graph(p) for p in graph_paths ]

def read_graph_task( graph_path ):
    graph = read_graph( graph_path )
    return (graph_path, graph)

def read_graphs_parallel( graph_paths, return_sorted=False ):
    n_cpus = psutil.cpu_count( logical=True )
    workers = mp.Pool( n_cpus )
    res = workers.map_async( read_graph_task, graph_paths )
    graphs = res.get()
    workers.close()
    if return_sorted:
        return [ g[1] for g in sorted( graphs, key=lambda x : x[0] ) ]
    else:
        return [ g[1] for g in graphs ]


# A function to read in a set of graphs whose paths are listed in a text file
@timer
def read_graphs( graph_list ):
    with open(graph_list, "r") as infile:
        paths = infile.readlines()
        paths = [ p.strip() for p in paths ]
        # Ignore empty lines if they're in the graph list by accident
        paths = list(filter(lambda line: len(line) > 0, paths))
    graphs = []
    for path in paths:
        graph = read_graph( path )
        graphs.append( graph )
    return graphs

def merge_dicts( list_of_dicts, check_keys=False ):
    merged = {}
    if check_keys:
        if not all_unique_keys( list_of_dicts ):
            raise RuntimeError("Duplicate keys present")
    for d in list_of_dicts:
        merged.update( d )
    return merged

def all_unique_keys( list_of_dicts ):
    all_keys = []
    for d in list_of_dicts:
        all_keys += list( d.keys() )
    key_set = set( all_keys )
    if len(key_set) != len(all_keys):
        return False
    else:
        return True
        
