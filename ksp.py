import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fmin_tnc
import igraph
import heapq

def MinHeaps_for_Vertices(Network, Source, edge_weights = False):

    '''
    Helper function for Step 1 of the Pathfind Algorithm.
    Construct a seperate min-heap for each node x in the network,
    where we weight the incoming neighbors, y, of x depending
    on their distance to the source.
    -------
    Inputs
            Network:      An object belonging to igraph's graph class.
                          The network contains labels for each of the nodes,
                          accessible through Network.vs['label'].

            Source:       The source for computing the almost shortest paths.

            edge_weights: If true, the graph is positively weighted and the
                          weights are accessible through the Network object

    Outputs
            heap_list:    a list of heaps for each node in the network.
    '''

    #if edge weights are provided
    if edge_weights:
        Heap_List = []
        Distances_to_Source = Get_List(Network.shortest_paths_dijkstra(source = Source))
        for node in Network.vs['label']:
            #identify incoming neighbors of node
            Neighbors = Network.neighbors(node, mode = 'in')
            Tuple_List = []

            for neighbor in Neighbors:
                #get edge weight
                eid = Network.get_eid(neighbor,node)
                edge = Network.es[eid]
                weight = edge['weight']
                Tuple_List.append((Distances_to_Source[neighbor] + weight, neighbor))
            heapq.heapify(Tuple_List)
            Heap_List.append(Tuple_List)

    #otherwise assume each edge has a weight of 1
    else:
        Heap_List = []
        Distances_to_Source = Get_List(Network.shortest_paths_dijkstra(source = Source))
        for node in Network.vs['label']:
            #identify incoming neighbors of node
            Neighbors = Network.neighbors(node, mode = 'in')
            Tuple_List = [(Distances_to_Source[neighbor] + 1,neighbor)
                      for neighbor in Neighbors]
            heapq.heapify(Tuple_List)
            Heap_List.append(Tuple_List)

    return Heap_List

class Path_Tree(object):
    """Implementation of the class for Step 2 of the PathFind algorithm
    to create a path tree, where a given object has Parent and Children
    attributes to navigate the tree.  In addition, the ID attribute
    maps nodes from the Path Tree to the original network and the
    trackdistance attribute tracks the distance of the corresponding
    path in the original network.  See paper for motivation.
    """

    def __init__(self):
        self.Parent = False
        self.ID = False
        self.Children = []
        self.TrackDistance = 0

    def Set_ID(self, id_value):
        self.ID = id_value

    def Add_Child(self,New_Child):
        self.Children.append(New_Child)

    def Set_Distance(self,distance):
        self.TrackDistance = distance

    def Set_Parent(self,New_Parent):
        self.Parent = New_Parent

def Heap_Search(Heap,MaxValue):
    '''
    Helper function for Step 3 of the Pathfind Algorithm.
    Find all nodes in a heap that are bounded above by a prescribed value (MaxValue).
    -------
    Inputs
            Heap:         A minheap created using the Minheaps_for_Vertices function.

            MaxValue:     The cutoff for returning all nodes in the minheap with a
                          key bounded above by MaxValue.
    Outputs
            Satisfy_Criteria: All nodes that satisfy that criteria.
    '''

    Satisfy_Criteria = []
    Queue = []
    Queue.append((0,Heap[0]))
    while Queue != []:
        nested_tuple = Queue.pop()
        heap_index = nested_tuple[0]
        pair = nested_tuple[1]
        distance = pair[0]
        if distance <= MaxValue:
            Satisfy_Criteria.append(pair)
            #add children to queue to check to see if they satisfy condition
            if 2*(heap_index + 1) - 1 < len(Heap):
                new_index = 2*(heap_index + 1) - 1
                Queue.append((new_index, Heap[new_index]))
            if 2*(heap_index + 1) + 1 - 1 < len(Heap):
                new_index = 2*(heap_index + 1) + 1 - 1
                Queue.append((new_index, Heap[new_index]))

    return Satisfy_Criteria

def PathFind(Source,Target,Network,max_length,edge_weights = False):
    '''
    Outputs all paths from a source to a target with length at most max_length.
    See paper for explanation of the algorithm.
    -------
    Inputs
            Source:      The source of the path.

            Target:      The target of the path.

            Network:      An object belonging to igraph's graph class.
                          The network contains labels for each of the nodes,
                          accessible through Network.vs['label'].

            max_length:   The upperbound for the length of all paths returned.

            edge_weights: If true, the graph is positively weighted and the
                          weights are accessible through the Network object

    Outputs
            Path_List:    a list of all paths from source to target with length at most max_length.
    '''


    #Step 1: Create a minheap for each node in network.
    Heap_List = MinHeaps_for_Vertices(Network, Source, edge_weights)

    #Step 2: Initialize path tree
    Root = Path_Tree()
    Root.Set_ID(Target)
    Root.Set_Distance(0)
    #Path_Start keeps track of which nodes in the path tree correspond
    #to the beginning of the path in the original network
    Path_Start = []
    Queue = []
    Queue.append(Root)

    #Step 3: Construct Path Tree
    while Queue != []:
        PathTreeParent = Queue.pop()
        Node_in_Network = PathTreeParent.ID
        Heap = Heap_List[Node_in_Network]
        Close_Nodes_to_Source = Heap_Search(Heap,max_length - PathTreeParent.TrackDistance)

        for pair in Close_Nodes_to_Source:
            node = pair[1]
            distance = pair[0]
            New_PathTreeChild = Path_Tree()
            New_PathTreeChild.Set_ID(node)
            if edge_weights:
                #weight = edge_weights[(node,PathTreeParent.ID)]
                eid = Network.get_eid(node, PathTreeParent.ID)
                edge = Network.es[eid]
                weight = edge['weight']
            else:
                weight = 1
            New_PathTreeChild.Set_Distance(PathTreeParent.TrackDistance + weight)
            New_PathTreeChild.Set_Parent(PathTreeParent)
            PathTreeParent.Add_Child(New_PathTreeChild)
            Queue.append(New_PathTreeChild)

            if node == Source:
                Path_Start.append(New_PathTreeChild)

    #Step 4: Construct Path List
    Path_List = []
    for PathTreeNode in Path_Start:
        Valid_Path = [ ]
        CurrentNode = PathTreeNode
        while CurrentNode.Parent:
            Valid_Path.append(CurrentNode.ID)
            CurrentNode = CurrentNode.Parent
        #once you reach root, the loop stops, so we need to add the root to the path
        Valid_Path.append(CurrentNode.ID)
        Path_List.append(Valid_Path)

    return Path_List

def CountSimplePaths(List_of_Paths):
    '''
    Count simple paths from a list of paths
    -------
    Inputs
            List_of_Paths: A list of paths.

    Outputs
            SimplePathCount: Number of simple paths in list of paths.
    '''
    SimplePathCount = 0
    for path in List_of_Paths:
        if len(path) == len(set(path)):
            #all nodes are distinct
            SimplePathCount += 1
    return SimplePathCount

def CountNonBacktrackingPaths(List_of_Paths):
    '''
    Count nonbacktracking paths from a list of paths
    -------
    Inputs
            List_of_Paths: A list of paths.

    Outputs
            NBPPathCount: Number of nonbacktracking paths in list of paths.
    '''
    BPCount = 0
    for path in List_of_Paths:
        for index in range(len(path) - 2):
            if path[index] == path[index + 2]:
                BPCount += 1
                break

    NBPCount = len(List_of_Paths) - BPCount
    return NBPCount

def GetNonBacktrackingPaths(List_of_Paths):
    '''
    Get all nonbacktracking paths from a list of paths
    -------
    Inputs
            List_of_Paths: A list of paths.

    Outputs
            Answer: The nonbacktracking paths in list of paths.
    '''
    Answer = []
    for path in List_of_Paths:
        BP = False
        for index in range(len(path) - 2):
            if path[index] == path[index + 2]:
                BP = True
        if not BP:
            Answer.append(path)

    return Answer

def GetSimplePaths(List_of_Paths):
    '''
    Get all simple paths from a list of paths
    -------
    Inputs
            List_of_Paths: A list of paths.

    Outputs
            Answer: The simple paths in list of paths.
    '''
    Answer = []
    for path in List_of_Paths:
        if len(path) == len(set(path)):
            Answer.append(path)

    return Answer

def Get_List(List_of_Lists):

    return List_of_Lists[0]
