from ase.io import read, write
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
import pickle
from ase.parallel import paropen

energies_slab = {}
for name in ['BiCl2', 'BiS2', 'BiSCl']:
    structures = name+'.xyz'
    structure = read(structures)
    
    calc = GPAW(mode=PW(500),
                xc='RPBE',
                kpts={'density': 3, 'gamma': True},
                poissonsolver={'dipolelayer': 'xy'},
                txt=name+'.out',
                )
    structure.pbc = (True, True, False)
    structure.set_calculator(calc)
    unitcell = UnitCellFilter(structure)
    relax = BFGS(unitcell, trajectory='relax--slab-'+name+'.traj', logfile='relax-slab-'+name+'.log')
    relax.run(fmax=0.1)
    energies_slab[name] = structure.get_potential_energy()
    calc.write('relax-'+name+'.gpw')  
    
#save
with paropen('energies_slab.pckl', 'wb') as f:  #binary protocols output
    pickle.dump(energies_slab, f)
    
with paropen('energies_slab.txt', 'w') as f:  # formatting output
    for x, y in energies_slab.items():
        print('energy-{0:6}: {1:6.3f}'.format(x, y), file=f)
