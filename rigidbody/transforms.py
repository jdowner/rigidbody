import numpy as np

from numbers import Number
from typing import (Optional, TypeVar, Union)

__all__ = ['Matrix', 'Vector', 'Rotation', 'Translation', 'Transform']


TranslationType = TypeVar("D", bound="Translation")
TransformType = TypeVar("T", bound="Transform")
RotationType = TypeVar("R", bound="Rotation")
MatrixType = TypeVar("M", bound="Matrix")
VectorType = TypeVar("V", bound="Vector")


class MatrixSizeError(Exception):
    def __init__(self, a, b):
        super(MatrixSizeError, self).__init__(self.msg.format(
                "{}x{}".format(a.rows, a.cols),
                "{}x{}".format(b.rows, b.cols),
                ))


class MatrixMultiplySizeError(MatrixSizeError):
    msg = "tried to multiply a {} and {} matrix"


class MatrixSubtractionSizeError(MatrixSizeError):
    msg = "tried to subtract a {} matrix from a {} matrix"


class MatrixAdditionSizeError(MatrixSizeError):
    msg = "tried to add a {} matrix from a {} matrix"


class Matrix(object):

    def __init__(self, rows, cols):

        self._rows = rows
        self._cols = cols
        self._data = np.zeros((rows, cols))

    @classmethod
    def zero(cls, rows, cols=1):
        m = cls(rows, cols)
        m._data = np.zeros((rows, cols))
        return m

    @classmethod
    def identity(cls, rows: int, cols: Optional[int]=None):
        if cols is None:
            cols = rows

        m = cls(rows, cols)
        m._data = np.eye(rows, cols)
        return m

    def __str__(self):
        return str(self._data)

    def __add__(self, other: MatrixType):

        if self.size != other.size:
            raise MatrixAdditionSizeError(self, other)

        m = Matrix(self.rows, self.cols)
        m._data = self._data + other._data

        return m

    def __iadd__(self, other: MatrixType):

        if self.size != other.size:
            raise MatrixAdditionSizeError(self, other)

        self._data += other._data

        return self

    def __sub__(self, other: MatrixType):

        if self.size != other.size:
            raise MatrixSubtractionSizeError(self, other)

        m = Matrix(self.rows, self.cols)
        m._data = self._data - other._data

        return m

    def __isub__(self, other: MatrixType):

        if self.size != other.size:
            raise MatrixSubtractionSizeError(self, other)

        self._data -= other._data

        return self

    def __mul__(self, other : Union[Number, MatrixType]):

        if self.cols != other.rows:
            raise MatrixMultiplySizeError(self, other)

        if isinstance(other, Number):

            m = Matrix(self.rows, self.cols)
            m._data = other * np.array(self._data)

            return m

        elif isinstance(other, Matrix):

            assert(self.cols == other.rows)

            m = Matrix(self.rows, other.cols)
            m._data = np.matmul(self._data, other._data)

            return m

    def __rmul__(self, other: Number):

        m = Matrix(self.rows, self.cols)
        m._data = other * np.array(self._data)

        return m

    def __eq__(self, other: MatrixType):

        return np.all(self._data == other._data)

    def __getitem__(self, range: tuple):
        return self._data.__getitem__(range)

    def __setitem__(self, range: tuple, value):
        return self._data.__setitem__(range, value)

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def size(self):
        return (self.rows, self.cols)

    @property
    def data(self):
        return self._data

    def transpose(self):
        self._data.transpose()

    def transposed(self):

        m = Matrix(self.cols, self.rows)
        m._data = np.array(self._data)
        m._data.transpose()

        return m

    def norm(self):
        return np.linalg.norm(self._data)


class Vector(Matrix):

    def __init__(self, *values):

        super(Vector, self).__init__(len(values), 1)

        self._data[:,0] = values

    def normalize(self):

        assert(self.norm() != 0)

        self._data /= self.norm()


class Rotation(object):

    AXIS_X = Vector(1, 0, 0)
    AXIS_Y = Vector(0, 1, 0)
    AXIS_Z = Vector(0, 0, 1)

    def __init__(self, rotation: Optional[MatrixType]=Matrix.identity(3)):

        self._rotation = rotation

    @classmethod
    def axis_angle(cls, axis : Union[tuple, Vector], angle : Number):

        if isinstance(axis, (list, tuple)):
            assert(len(axis) == 3)
            axis = Vector(*axis)

        assert(isinstance(axis, Vector))

        K = Matrix(3, 3)

        axis.normalize()

        x = axis[0, 0]
        y = axis[1, 0]
        z = axis[2, 0]

        K[0, 1] = -z
        K[0, 2] = y
        K[1, 2] = -x

        K[1, 0] = z
        K[2, 0] = -y
        K[2, 1] = x

        c = np.cos(angle)
        s = np.sin(angle)

        I = Matrix.identity(3)

        rot = I + s * K + (1 - c) * K * K

        return cls(rot)

    @classmethod
    def euler_angles(cls, roll, pitch, yaw):
        pass

    def __mul__(self, other: Union[RotationType, TranslationType]):

        if isinstance(other, Rotation):
            r = Rotation()
            r._rotation = self.matrix * other.matrix
            return r

        elif isinstance(other, Translation):
            t = Translation()
            t._translation = self.matrix * other.matrix
            return t

    def __imul__(self, other: RotationType):

        self._rotation *= other._rotation

        return self

    def invert(self):

        self._rotation.transpose()

    def inverse(self):

        return Rotation(self.matrix.transposed())

    @property
    def matrix(self):

        return self._rotation


class Translation(object):

    def __init__(self, x=0, y=0, z=0):

        self._translation = Vector(x, y, z)

    def __str__(self):
        return str(self._translation)

    @property
    def x(self):
        return self._translation[0, 0]

    @x.setter
    def x(self, value):
        self._translation[0, 0] = value

    @property
    def y(self):
        return self._translation[1, 0]

    @y.setter
    def y(self, value):
        self._translation[1, 0] = value

    @property
    def z(self):
        return self._translation[2, 0]

    @z.setter
    def z(self, value):
        self._translation[2, 0] = value

    def __add__(self, other: TranslationType):

        return Translation(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z)

    def __neg__(self):

        return Translation(-self.x, -self.y, -self.z)

    def __sub__(self, other: TranslationType):

        return Translation(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z)

    def __mul__(self, value: Number):

        return Translation(
                value * self.x,
                value * self.y,
                value * self.z)

    def __rmul__(self, value: Number):

        return Translation(
                value * self.x,
                value * self.y,
                value * self.z)

    @property
    def matrix(self):
        return self._translation


class Transform(object):

    def __init__(self, rotation: RotationType, translation: TranslationType):

        self._rotation = rotation
        self._translation = translation

    def __str__(self):
        m = Matrix.identity(4, 4)

        m[:3, :3] = self.rotation.matrix.data
        m[:3, 3:4] = self.translation.matrix.data

        return str(m)

    @property
    def rotation(self):
        return self._rotation

    @property
    def translation(self):
        return self._translation

    def __mul__(self, other : TransformType):

        return Transform(
                self.rotation * other.rotation,
                self.rotation * other.translation + self.translation)

    def inverse(self):

        rotation = self.rotation.inverse()
        translation = rotation * (-self.translation)

        return Transform(rotation, translation)
