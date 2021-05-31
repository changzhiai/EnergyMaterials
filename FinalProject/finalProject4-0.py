from ase.io import read, write
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
import pickle
from ase.parallel import paropen

energies_tube_slab = {}
structures = 'BiSCl_1D.xyz'
structure = read('structures')

calc = GPAW(mode=PW(500),
            xc='RPBE',
            kpts={'density': 3, 'gamma': True},
            #poissonsolver={'dipolelayer': 'xy'},
            txt='out-BiSCl_1D.out',
            )
structure.pbc = (False, False, True)
structure.set_calculator(calc)
#unitcell = UnitCellFilter(structure)
relax = BFGS(structure, trajectory='relax--slab-BiSCl_1D.traj', logfile='log-relax-slab-BiSCl_1D.log')
relax.run(fmax=0.1)
energies_tube_slab['BiSCl_1D'] = structure.get_potential_energy()
calc.write('gpw-relax-BiSCl_1D.gpw')  # We are going to need this gpw file later
#write('opti-BiClI.xyz', read('relax-BiClI.xyz.traj'))
    
#print(energies_slab)
with paropen('energies_tube_slab.pckl', 'wb') as f:  #binary protocols output
    pickle.dump(energies_tube_slab, f)
    
with paropen('energies_tube_slab.txt', 'w') as f:  # formatting output
    for x, y in energies_tube_slab.items():
        print('energy-{0:6}: {1:6.3f}'.format(x, y), file=f)
