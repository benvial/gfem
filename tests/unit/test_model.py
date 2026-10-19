"""`gfem.Model`: an immutable, validated, problem-independent device description.

Fixture (see test_mesh.py): "sub" holds a dielectric entity and a 1 x 1 x 0.4
um conductor entity that gsim-style tagging also names "sub"; the conductor's
shell is "metal_xy" + "metal_z". "air" sits on top; "air__sub" is their
interface; "sub__None" and "air__None" are the exterior.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

DATA = Path(__file__).parent / "data"
SI = gfem.Material(eps_r=11.7, name="si")
AIR = gfem.Material(name="air")
METAL = gfem.enclosed_by("metal_xy", "metal_z")


@pytest.fixture(params=["msh22", "msh41"])
def mesh(request):
    return gfem.Mesh.from_gmsh(DATA / f"two_box_{request.param}.msh")


def _conductor_entity(mesh):
    """The gmsh entity of "sub" whose volume is the 0.4 um^3 conductor box."""
    x = np.asarray(mesh.nodes)[mesh.tetrahedra]
    vol = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6
    sub = mesh.elements("sub")
    ents = np.unique(mesh.tetrahedron_entities[sub])
    sizes = {int(e): vol[sub][mesh.tetrahedron_entities[sub] == e].sum() for e in ents}
    (entity,) = [e for e, v in sizes.items() if abs(v - 0.4e-18) < 1e-30]
    return entity


def test_model_holds_materials_boundaries_and_ports(mesh):
    port = gfem.LumpedPort("p1", "air__sub", direction="+z")
    model = gfem.Model(
        mesh,
        materials={"air": AIR, "sub": SI},
        boundaries=[gfem.Absorbing("air__None")],
        ports=[port],
    )
    assert model.mesh is mesh
    assert dict(model.materials) == {"sub": SI, "air": AIR}
    assert list(model.materials) == ["sub", "air"]  # mesh volume order
    assert model.boundaries == (gfem.Absorbing("air__None"),)
    assert model.ports == (port,)
    assert model.port_names == ("p1",)
    assert model.excluded == ()


def test_model_is_immutable(mesh):
    model = gfem.Model(mesh, materials={"sub": SI, "air": AIR})
    with pytest.raises(AttributeError):
        model.mesh = mesh  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        model.materials["air"] = SI  # ty: ignore[invalid-assignment]


def test_unknown_material_region_suggests_a_close_match(mesh):
    with pytest.raises(gfem.RegionNotFoundError, match="Did you mean 'air'"):
        gfem.Model(mesh, materials={"sub": SI, "ari": AIR})


def test_material_on_a_surface_says_it_is_a_surface(mesh):
    with pytest.raises(ValueError, match="'metal_xy' is a surface"):
        gfem.Model(mesh, materials={"sub": SI, "air": AIR, "metal_xy": AIR})


def test_every_volume_needs_a_material_or_exclusion(mesh):
    with pytest.raises(ValueError, match=r"'air'.*excluded="):
        gfem.Model(mesh, materials={"sub": SI})


def test_materials_must_be_material_values(mesh):
    with pytest.raises(TypeError, match="Material or MaterialField"):
        gfem.Model(mesh, materials={"sub": SI, "air": 1.0})


# --- excluded regions -----------------------------------------------------------


def test_excluding_a_region_by_name_removes_all_its_entities(mesh):
    model = gfem.Model(mesh, materials={"air": AIR}, excluded=("sub",))
    ents = sorted({int(e) for e in mesh.tetrahedron_entities[mesh.elements("sub")]})
    assert model.excluded_entities == tuple(("sub", e) for e in ents)


def test_an_excluded_region_cannot_also_have_a_material(mesh):
    with pytest.raises(ValueError, match="excluded"):
        gfem.Model(mesh, materials={"air": AIR, "sub": SI}, excluded=("sub",))


def test_excluded_names_must_be_volumes(mesh):
    with pytest.raises(gfem.RegionNotFoundError, match="Did you mean 'sub'"):
        gfem.Model(mesh, materials={"air": AIR}, excluded=("sbu",))
    with pytest.raises(ValueError, match="enclosed_by"):
        gfem.Model(mesh, materials={"air": AIR, "sub": SI}, excluded=("metal_xy",))
    with pytest.raises(TypeError, match="enclosed_by"):
        gfem.Model(mesh, materials={"air": AIR, "sub": SI}, excluded=(3,))


def test_enclosed_by_excludes_the_conductor_entity_inside_its_dielectric(mesh):
    model = gfem.Model(mesh, materials={"sub": SI, "air": AIR}, excluded=(METAL,))
    assert model.excluded_entities == (("sub", _conductor_entity(mesh)),)
    assert model.materials["sub"] is SI  # the rest of "sub" keeps its material
    assert "enclosed_by('metal_xy', 'metal_z')" in repr(model)
    assert f"sub entity {_conductor_entity(mesh)}" in repr(model)


def test_enclosed_by_needs_the_whole_shell(mesh):
    with pytest.raises(ValueError, match="no volume entity"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            excluded=(gfem.enclosed_by("metal_xy"),),
        )


def test_enclosed_by_takes_surface_names(mesh):
    with pytest.raises(ValueError, match="at least one"):
        gfem.enclosed_by()
    with pytest.raises(gfem.RegionNotFoundError, match="metal_xy"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            excluded=(gfem.enclosed_by("metal_yx", "metal_z"),),
        )


def test_excluding_everything_is_an_error(mesh):
    with pytest.raises(ValueError, match="nothing left"):
        gfem.Model(mesh, materials={}, excluded=("sub", "air"))


# --- boundaries, ports and the PMC default --------------------------------------


def test_unassigned_exterior_surfaces_default_to_pmc(mesh):
    model = gfem.Model(mesh, materials={"sub": SI, "air": AIR})
    assert model.default_pmc == ("sub__None", "air__None")
    assert "PMC" in repr(model)
    assert "electromagnetic" in repr(model)


def test_conductor_shells_of_excluded_entities_are_exterior(mesh):
    model = gfem.Model(
        mesh,
        materials={"sub": SI, "air": AIR},
        boundaries=[gfem.Conductor("metal_xy", conductivity=5.8e7)],
        excluded=(METAL,),
    )
    assert model.default_pmc == ("metal_z", "sub__None", "air__None")


def test_boundaries_name_existing_surfaces(mesh):
    with pytest.raises(gfem.RegionNotFoundError, match="Did you mean 'air__None'"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            boundaries=[gfem.PEC("air_None")],
        )
    with pytest.raises(ValueError, match="'air' is a volume"):
        gfem.Model(
            mesh, materials={"sub": SI, "air": AIR}, boundaries=[gfem.PEC("air")]
        )
    with pytest.raises(TypeError, match="boundaries"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            boundaries=[gfem.LumpedPort("p", "air__sub", "+z")],
        )


def test_at_most_one_electromagnetic_condition_per_surface(mesh):
    with pytest.raises(ValueError, match="more than one electromagnetic"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            boundaries=[gfem.PEC("air__None"), gfem.Absorbing("air__None")],
        )
    with pytest.raises(ValueError, match="more than one electromagnetic"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            boundaries=[gfem.PEC("air__sub")],
            ports=[gfem.LumpedPort("p", "air__sub", "+z")],
        )


def test_port_names_are_unique(mesh):
    with pytest.raises(ValueError, match="'p'"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            ports=[
                gfem.LumpedPort("p", "air__sub", "+z"),
                gfem.LumpedPort("p", "sub__None", "+z"),
            ],
        )


def test_absorbing_and_wave_ports_need_exterior_surfaces(mesh):
    with pytest.raises(ValueError, match="exterior"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            boundaries=[gfem.Absorbing("air__sub")],
        )
    with pytest.raises(ValueError, match="exterior"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            ports=[gfem.WavePort("w", "air__sub")],
        )


def test_wave_port_pec_edges_name_surfaces(mesh):
    with pytest.raises(gfem.RegionNotFoundError, match="metal_z"):
        gfem.Model(
            mesh,
            materials={"sub": SI, "air": AIR},
            ports=[gfem.WavePort("w", "air__None", pec_edges=("metal_zz",))],
        )


def test_boundaries_must_touch_the_solve_domain(mesh):
    with pytest.raises(ValueError, match="solve domain"):
        gfem.Model(
            mesh,
            materials={"air": AIR},
            boundaries=[gfem.PEC("metal_xy")],
            excluded=("sub",),
        )


# --- material fields --------------------------------------------------------------


def test_material_field_has_one_value_per_region_element(mesh):
    n = len(mesh.elements("sub"))
    field = gfem.MaterialField(eps_r=jnp.full(n, 4.0))
    model = gfem.Model(mesh, materials={"sub": field, "air": AIR})
    assert model.materials["sub"] is field
    with pytest.raises(ValueError, match=f"{n} tetrahedra"):
        gfem.Model(
            mesh,
            materials={"sub": gfem.MaterialField(eps_r=jnp.ones(n + 1)), "air": AIR},
        )


def test_uniform_field_must_go_to_its_region(mesh):
    gfem.Model(
        mesh, materials={"sub": gfem.MaterialField.uniform("sub", SI), "air": AIR}
    )
    with pytest.raises(ValueError, match="built for region 'sub'"):
        gfem.Model(
            mesh, materials={"sub": SI, "air": gfem.MaterialField.uniform("sub", AIR)}
        )


# --- deriving models --------------------------------------------------------------


def test_replace_and_with_material_return_new_validated_models(mesh):
    model = gfem.Model(mesh, materials={"sub": SI, "air": AIR})
    other = model.with_material("air", SI)
    assert other.materials["air"] is SI
    assert model.materials["air"] is AIR
    absorbing = model.replace(boundaries=[gfem.Absorbing("air__None", order=2)])
    assert absorbing.boundaries == (gfem.Absorbing("air__None", order=2),)
    assert absorbing.default_pmc == ("sub__None",)
    assert model.boundaries == ()
    with pytest.raises(gfem.RegionNotFoundError):
        model.with_material("aier", SI)
    with pytest.raises(TypeError, match="colour"):
        model.replace(colour="red")
    with pytest.raises(ValueError, match="excluded"):
        model.replace(excluded=("air",))


def test_moved_mesh_keeps_the_excluded_entities(mesh):
    model = gfem.Model(mesh, materials={"sub": SI, "air": AIR}, excluded=(METAL,))
    moved = jax.tree.map(lambda x: x * 1.01, mesh)
    assert model.replace(mesh=moved).excluded_entities == model.excluded_entities


# --- pytree -----------------------------------------------------------------------


def test_model_is_a_pytree_of_its_differentiable_arrays(mesh):
    n = len(mesh.elements("sub"))
    field = gfem.MaterialField(eps_r=jnp.full(n, 4.0), conductivity=jnp.zeros(n))
    model = gfem.Model(
        mesh,
        materials={"sub": field, "air": AIR},
        boundaries=[gfem.Conductor("metal_xy", conductivity=1e7)],
        excluded=(METAL,),
    )
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(model)
    ]
    assert paths == [
        ".mesh.nodes",
        ".materials['sub'].eps_r",
        ".materials['sub'].conductivity",
    ]
    doubled = jax.tree.map(lambda x: 2 * x, model)
    assert isinstance(doubled, gfem.Model)
    assert list(doubled.materials) == ["sub", "air"]
    assert doubled.excluded_entities == model.excluded_entities
    np.testing.assert_allclose(doubled.materials["sub"].eps_r, 8.0)

    def objective(m):
        return jnp.sum(m.materials["sub"].eps_r ** 2) + jnp.sum(m.mesh.nodes)

    grads = jax.grad(objective)(model)
    np.testing.assert_allclose(grads.materials["sub"].eps_r, 8.0)
    np.testing.assert_allclose(grads.mesh.nodes, 1.0)


def test_model_passes_through_jit(mesh):
    n = len(mesh.elements("sub"))
    model = gfem.Model(
        mesh,
        materials={"sub": gfem.MaterialField(eps_r=jnp.full(n, 4.0)), "air": AIR},
    )
    total = jax.jit(lambda m: jnp.sum(m.materials["sub"].eps_r))(model)
    assert float(total) == 4.0 * n


def test_a_model_built_inside_jax_grad_carries_the_gradient(mesh):
    base = gfem.Model(mesh, materials={"sub": SI, "air": AIR}, excluded=(METAL,))
    n = len(mesh.elements("sub"))

    def objective(rho):
        field = gfem.MaterialField.interpolate(rho, AIR, SI)
        return jnp.sum(base.with_material("sub", field).materials["sub"].eps_r)

    grad = jax.grad(objective)(jnp.full(n, 0.5))
    np.testing.assert_allclose(grad, 10.7)


def test_model_needs_a_mesh():
    with pytest.raises(TypeError, match=r"gfem\.Mesh"):
        gfem.Model("two_box.msh", materials={})  # ty: ignore[invalid-argument-type]


# --- reference meshes ----------------------------------------------------------------


def _driven_cases():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        return [pytest.param(None, id="no-reference-dir")]
    return [
        pytest.param(Path(ref) / "driven" / c, id=c) for c in ("r8", "r4", "r8_box150")
    ]


@pytest.mark.parametrize("case", _driven_cases())
def test_enclosed_by_finds_the_gsim_conductors_inside_sio2(case):
    if case is None:
        pytest.skip("GFEM_REFERENCE_DIR not set (point it at the reference data)")
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh")
    model = gfem.Model(
        mesh,
        materials={
            "sio2": gfem.Material(eps_r=4.1),
            "sin": gfem.Material(eps_r=7.5, tan_delta=1e-3),
            "air": gfem.Material(),
        },
        excluded=(gfem.enclosed_by("topmetal2_xy", "topmetal2_z"),),
    )
    assert model.excluded_entities == (("sio2", 5), ("sio2", 6), ("sio2", 7))
