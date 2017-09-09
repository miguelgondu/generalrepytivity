import sympy

class(object) VectorField:
    def __init__(self, _matrix, _coordinates):
        self.matrix = _matrix # Assumed to be vertical nx1
        self.coordinates = _coordinates

    def change_coordinates(self, new_coordinates, relation_between_coordinates):
        '''
        This function changes the vector fields from one base
        to the other.

        To-Do:
            -Implement it
        '''
        return

class(object) Metric:
    def __init__(self, _matrix, _coordinates):
        self.matrix = _matrix
        self.coordinates = _coordinates
    
    def distance(vector1, vector2):
        '''
        This function computes the distance between two vectors according
        to the metric.

        To-Do:
            -Test this beauty.
        '''
        return (vector1.matrix.T * self.matrix) * vector2.matrix

