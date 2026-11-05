"""Points placed by hand near a tetrahedron's face, for tests of `Field.at`.

Each point's holder is known by construction, with no point search: a point at a face's
centroid moved ``delta`` towards a tet's opposite vertex has barycentric coordinate
``delta`` for that vertex and ``(1 - delta) / 3`` for the others.
"""

import numpy as np


def interior_faces(mesh):
    """``(n, 2)`` tets sharing a face, and ``(n, 3)`` the face's nodes."""
    seen, pairs, faces = {}, [], []
    for t, tet in enumerate(np.asarray(mesh.tetrahedra).tolist()):
        for skip in range(4):
            face = tuple(sorted(tet[:skip] + tet[skip + 1 :]))
            other = seen.setdefault(face, t)
            if other != t:
                pairs.append((other, t))
                faces.append(face)
    return np.array(pairs), np.array(faces)


def into(mesh, tets, faces, delta):
    """``(n, 3)`` points at ``faces``' centroids, ``delta`` into their tets ``tets``.

    Returns the points and ``(n, 4)`` their barycentric coordinates in ``tets``.
    """
    nodes = np.asarray(mesh.nodes)
    vertices = np.asarray(mesh.tetrahedra)[tets]
    opposite = (vertices[:, :, None] != faces[:, None, :]).all(axis=2)
    bary = np.where(opposite, delta, (1.0 - delta) / 3.0)
    return (nodes[vertices] * bary[:, :, None]).sum(axis=1), bary
