############## GLOBAL YIELDS -- MODIFY BELOW ##############

# The pre-set values here are the yields we adopt in the
# Multi-zone section of Weller, Weinberg, and Johnson (2024)

helium_agb = 'ventura13' # allowed values: 'ventura13', 'cristallo11', 'karakas10', 'karakas16'
nitrogen_agb = 'cristallo11' # allowed values: 'ventura13', 'cristallo11', 'karakas10', 'karakas16'
nitrogen_agb_prefactor = 'oxygen_scaled' # 'oxygen_scaled' or a float
carbon_agb = 'cristallo11' # allowed values: 'ventura13', 'cristallo11', 'karakas10', 'karakas16'
carbon_agb_prefactor = 1.36 # allowed values: float

helium_ccsne = 'nkt13_ext' # allowed values: 'lc18', 'nkt13', 'nkt13_ext'
ccsne_rot = 0 # allowed values: 0, 150, 300
net = True # allowed values: True or False
oxygen_ccsne = 'weinberg23_enhanced' # allowed values: 'weinberg23_enhanced', 'weinberg23_default', or a float
iron_ccsne = 'weinberg23_enhanced' # allowed values: 'weinberg23_enhanced', 'weinberg23_default', or a float
nitrogen_ccsne = 'johnson23' # allowed values: 'johnson23', function of z, or a float
carbon_ccsne = lambda z: 2.28*10**(-3) + 5.28e-2 * (z - 0.014) # allowed values: function of z or a float

iron_ia = 'weinberg23_enhanced' # allowed values: 'weinberg23_enhanced', 'weinberg23_default', or a float

############## GLOBAL YIELDS -- MODIFY ABOVE ##############

############## DO NOT MODIFY BELOW THIS LINE ##############
############## DO NOT MODIFY BELOW THIS LINE ##############
############## DO NOT MODIFY BELOW THIS LINE ##############
############## DO NOT MODIFY BELOW THIS LINE ##############

import warnings
warnings.filterwarnings("ignore")
import vice
import pandas as pd
import numpy as np
import scipy as sc
import scipy.integrate as integrate
import numbers

SolarO = 7.33e-3
SolarMg = 6.71e-4
SolarFe = 1.37e-3

vice.solar_z['o'] = SolarO
vice.solar_z['mg'] = SolarMg
vice.solar_z['fe'] = SolarFe

if type(oxygen_ccsne) != type(iron_ia) or type(oxygen_ccsne) != type(iron_ccsne) or type(iron_ia) != type(iron_ccsne):
    raise Exception("If any variation of yields based off Weinberg++23 is selected, they must match for CCSNe Oxygen, Iron and SNeIa Iron.")

if isinstance(oxygen_ccsne, str):
    if oxygen_ccsne != iron_ccsne or oxygen_ccsne != iron_ia or iron_ccsne != iron_ia:
        raise Exception("If any variation of yields based off Weinberg++23 is selected, they must match for CCSNe Oxygen, Iron and SNeIa Iron.")

if oxygen_ccsne == 'weinberg23_enhanced' and iron_ccsne == 'weinberg23_enhanced' and iron_ia == 'weinberg23_enhanced':
    # IMF-averaged CCSN yields
    # yield calibration is based on Weinberg++ 2023, eq. 11
    afecc = 0.45              # plateau value for [alpha/Fe]
    mocc = 0.973 * SolarO * (0.00137/SolarFe) * (10**(afecc - 0.45)) * 10**(0.1)  # CCSN oxygen
    mfecc = mocc * (SolarFe/SolarO) * (10**(-afecc))         # CCSN iron

    # population averaged SNIa Fe yield, integrated to t=infty
    # for a constant SFR, will evolve to afeeq at late times
    afeeq = 0.05
    mfeIa = mfecc*(10.**(afecc-afeeq) - 1.)

elif oxygen_ccsne == 'weinberg23_default' and iron_ccsne == 'weinberg23_default' and iron_ia == 'weinberg23_default':
    # IMF-averaged CCSN yields
    # yield calibration is based on Weinberg++ 2023, eq. 11
    afecc = 0.45              # plateau value for [alpha/Fe]
    mocc = 0.973 * SolarO * (0.00137/SolarFe) * (10**(afecc - 0.45))  # CCSN oxygen
    mfecc = mocc * (SolarFe/SolarO) * (10**(-afecc))         # CCSN iron

    # population averaged SNIa Fe yield, integrated to t=infty
    # for a constant SFR, will evolve to afeeq at late times
    afeeq = 0.05
    mfeIa = mfecc*(10.**(afecc-afeeq) - 1.)

else:
    mocc = oxygen_ccsne
    mfecc = iron_ccsne
    mfeIa = iron_ia

class amplified_agb(vice.yields.agb.interpolator):

	def __init__(self, element, study = "cristallo11", prefactor = 1):
		super().__init__(element, study = study)
		self.prefactor = prefactor

	def __call__(self, mass, metallicity):
		return self.prefactor * super().__call__(mass, metallicity)

	@property
	def prefactor(self):
		r"""
		Type : float

		Default : 1

		The multiplicative factor by which the yields are amplified. Must be
		non-negative.
		"""
		return self._prefactor

	@prefactor.setter
	def prefactor(self, value):
		if isinstance(value, numbers.Number):
			if value >= 0:
				self._prefactor = float(value)
			else:
				raise ValueError("Prefactor must be non-negative.")
		else:
			raise TypeError("Prefactor must be a numerical value. Got: %s" % (
				type(value)))

class ccsne_yields:

    def __init__(self, study, MoverH, remnants, birth, upper, lower, net = True, imf = 'False', rot = 0):

        self.study = study
        self.MoverH = MoverH
        self.remnants = remnants
        self.birth = birth
        self.upper = upper
        self.lower = lower
        self.cap = study.upper()
        self.rot = rot
        self.net = net

        total = vice.yields.ccsne.table('he', study = self.cap, MoverH = self.MoverH, rotation = self.rot, wind = True, isotopic = True)
        windless = vice.yields.ccsne.table('he', study = self.cap, MoverH = self.MoverH, rotation = self.rot, wind = False, isotopic = True)
        wind = [total[masses]['he4'] - windless[masses]['he4'] for masses in total.masses]
        exp = [windless[masses]['he4'] for masses in total.masses]

        self.wind = wind
        self.exp = exp
        self.masses = total.masses

        if net:
            yields = [wind[i] + exp[i] - birth * (total.masses[i] - self.remnants[i]) for i in range(len(total.masses))]
            if imf:
                netimf = [yields[i] * vice.imf.kroupa(total.masses[i]) for i in range(len(total.masses)) if (total.masses[i] <= 120)]
                self.netimf = netimf
        else:
            yields = [wind[i] + exp[i] for i in range(len(total.masses))]
            if imf:
                grossimf = [yields[i] * vice.imf.kroupa(total.masses[i]) for i in range(len(total.masses)) if (total.masses[i] <= 120)]
                self.grossimf = grossimf

        hel = vice.toolkit.interpolation.interp_scheme_1d([8] + list(total.masses), [yields[0] * 8/total.masses[0]] + yields)

        self.hel = hel
        self.yields = yields

    def numerator(self, m, y):
        '''
        Calculate the numerator for the fractional {net} yield of CCSNe

        Parameters
        ----------

        mass : float
               the progenitor mass of a star

        yields : interpolation scheme
                 the massive star net yields

        Returns
        -------

        numerator : float
                    The numerator for the fractional {net} yield equation of CCSNe
        '''

        num = y(m) * vice.imf.kroupa(m)

        return(num)

    def denominator(self, m):
        '''
        Calculate the denominator for the fractional {net} yield of CCSNe

        Parameters
        ----------

        mass : float
               The progenitor mass of a star

        Returns
        -------

        denominator : float
                    The denominator for the fractional {net} yield equation of CCSNe
        '''

        denom = m * vice.imf.kroupa(m)

        return(denom)

    def __call__(self):

        y = np.array(self.yields)[np.where((np.array(self.masses) >= self.lower) & (np.array(self.masses) <= self.upper))[0]]
        mto8 = np.array(self.masses)[np.where((np.array(self.masses) >= self.lower) & (np.array(self.masses) <= self.upper))[0]]

        if self.lower == 8:

            y = [self.yields[0] * 8/self.masses[0]] + list(y)
            mto8 = [8] + list(mto8)

        imf = [y[i] * vice.imf.kroupa(mto8[i]) for i in range(len(mto8))]
        ycc = integrate.trapezoid(imf, x = mto8)/integrate.quad(self.denominator, 0.08, 120)[0]

        return(ycc)

class lc18_interp:

    def __init__(self, birth, mupper, mlower, net = True, imf = False, rot = 0):

        self.birth = birth
        self.mupper = mupper
        self.mlower = mlower
        self.solar_z = 0.014
        self.rot = rot
        self.net = net
        self.imf = imf

        try:
            setR = pd.read_csv('mcut-SetR.txt', comment='#', delimiter='\s+') # LC18 initial-final mass relation
        except:
            raise Exception("Must download required file for LC18.")
        r_0 = np.abs(setR[setR['[Fe/H]'] == 0].reset_index()['mcut'])
        r_1 = np.abs(setR[setR['[Fe/H]'] == -1].reset_index()['mcut'])
        r_2 = np.abs(setR[setR['[Fe/H]'] == -2].reset_index()['mcut'])
        r_3 = np.abs(setR[setR['[Fe/H]'] == -3].reset_index()['mcut'])

        ycc_0 = ccsne_yields('lc18', 0, r_0, self.birth[0], self.mupper, self.mlower, self.net, self.imf, self.rot)
        ycc_1 = ccsne_yields('lc18', -1, r_1, self.birth[1], self.mupper, self.mlower, self.net, self.imf, self.rot)
        ycc_2 = ccsne_yields('lc18', -2, r_2, self.birth[2], self.mupper, self.mlower, self.net, self.imf, self.rot)
        ycc_3 = ccsne_yields('lc18', -3, r_3, self.birth[3], self.mupper, self.mlower, self.net, self.imf, self.rot)

        xcoords = [-3, -2, -1, 0]
        z = [10**x * self.solar_z for x in xcoords]

        he = [ycc_3(), ycc_2(), ycc_1(), ycc_0()]

        interp = vice.toolkit.interpolation.interp_scheme_1d(xcoords, he)
        self.interp = interp

    def custom_yield(self, zval):

        mh = np.log10(zval/self.solar_z)

        return (self.interp(mh))

class nkt_interp:

    def __init__(self, mupper, mlower, rot = 0, net = True, addLC18 = True):

        self.mupper = mupper
        self.mlower = mlower
        self.solar_z = 0.014
        self.rot = rot
        self.addLC18 = addLC18
        self.net = net

        xcoords = [-1.15, -0.54, -0.24, 0.15, 0.55]
        z = [10**x * self.solar_z for x in xcoords]

        def nkt_net_gross(df, birth):

            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df[df['M'] >= 10]
            df = df.reset_index(drop = True)

            col_list = list(df)
            col_list.remove('M')
            col_list.remove('E')
            col_list.remove('Mrem')

            df[r'TotalEjecta'] = df[col_list].sum(axis=1)

            sum_pzm = []
            wind = []
            net = []
            gross = []
            net_imf = []
            gross_imf = []

            for i in range(len(df[r'TotalEjecta'])):

                sum_pzm.append(df[r'TotalEjecta'][i]/df['M'][i])
                wind.append((1 - df['Mrem'][i]/df['M'][i] - sum_pzm[-1]))
                net.append(df['He4'][i] - df[r'TotalEjecta'][i] * birth)
                gross.append(df['He4'][i] + wind[-1] * birth * df['M'][i])
                net_imf.append(net[-1] * vice.imf.kroupa(df['M'][i]))
                gross_imf.append(gross[-1] * vice.imf.kroupa(df['M'][i]))

            df['Wind'] = wind
            df['Net (4He)'] = net
            df['Gross (4He)'] = gross
            df['Net IMF'] = net_imf
            df['Gross IMF'] = gross_imf

            return(df)

        try:
            nkt = pd.read_csv('NKT13_Selection.txt', sep = '\s+', comment = '#', header = None)
        except:
            raise Exception("Must download required file for NKT13.")


        nkt_n1p15 = pd.DataFrame.transpose(nkt[0:86])
        nkt_n1p15 = nkt_net_gross(nkt_n1p15, 0.25)

        nkt_np54 = pd.DataFrame.transpose(nkt[86:172])
        nkt_np54 = nkt_net_gross(nkt_np54, 0.25)

        nkt_np24 = pd.DataFrame.transpose(nkt[172:258])
        nkt_np24 = nkt_net_gross(nkt_np24, 0.25)

        nkt_p15 = pd.DataFrame.transpose(nkt[258:344])
        nkt_p15 = nkt_net_gross(nkt_p15, 0.265)

        nkt_p55 = pd.DataFrame.transpose(nkt[344:430])
        nkt_p55 = nkt_net_gross(nkt_p55, 0.274)

        nkt_all = [nkt_n1p15, nkt_np54, nkt_np24, nkt_p15, nkt_p55]

        def denominator(m):

            denom = m * vice.imf.kroupa(m)

            return(denom)

        def yields_mass(i):

            if self.mlower == 8:

                masses = [8] + list(nkt_all[i]['M'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])

                if net:
                    yields = [0] + list(nkt_all[i]['Net IMF'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])
                else:
                    yields = [0] + list(nkt_all[i]['Gross IMF'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])

            else:

                masses = list(nkt_all[i]['M'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])

                if net:
                    yields = list(nkt_all[i]['Net IMF'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])
                else:
                    yields = list(nkt_all[i]['Gross IMF'][(nkt_all[i]['M'] >= self.mlower) & ((nkt_all[i]['M'] <= self.mupper))])

            if self.addLC18 and self.mupper > 40:

                setR = pd.read_csv('mcut-SetR.txt', comment='#', delimiter='\s+') # LC18 initial-final mass relation
                r_0 = list(setR[setR['[Fe/H]'] == 0].reset_index()['mcut'])
                r_1 = list(np.abs(setR[setR['[Fe/H]'] == -1].reset_index()['mcut']))

                lc18_rems = [r_1, r_1, r_0, r_0, r_0]
                lc18_births = [2.500e-01, 2.500e-01, 2.650e-01, 2.650e-01, 2.650e-01]
                lc18_coords = [-1, -1, 0, 0, 0]

                if net:
                    y = ccsne_yields('LC18', lc18_coords[i], lc18_rems[i], lc18_births[i], upper = self.mupper, lower = self.mlower, net = True, imf = True, rot = self.rot).netimf
                    m = ccsne_yields('LC18', lc18_coords[i], lc18_rems[i], lc18_births[i], upper = self.mupper, lower = self.mlower, net = True, imf = True, rot = self.rot).masses
                else:
                    y = ccsne_yields('LC18', lc18_coords[i], lc18_rems[i], lc18_births[i], upper = self.mupper, lower = self.mlower, net = False, imf = True, rot = self.rot).grossimf
                    m = ccsne_yields('LC18', lc18_coords[i], lc18_rems[i], lc18_births[i], upper = self.mupper, lower = self.mlower, net = False, imf = True, rot = self.rot).masses

                mups = m[-3:]
                yups = y[-3:]
                masses = masses + list(mups)
                yields = yields + list(yups)

            return(masses, yields)

        ycc_115 = integrate.trapezoid(yields_mass(0)[1], x = yields_mass(0)[0])/integrate.quad(denominator, 0.08, 120)[0]
        ycc_054 = integrate.trapezoid(yields_mass(1)[1], x = yields_mass(1)[0])/integrate.quad(denominator, 0.08, 120)[0]
        ycc_024 = integrate.trapezoid(yields_mass(2)[1], x = yields_mass(2)[0])/integrate.quad(denominator, 0.08, 120)[0]
        ycc_015 = integrate.trapezoid(yields_mass(3)[1], x = yields_mass(3)[0])/integrate.quad(denominator, 0.08, 120)[0]
        ycc_055 = integrate.trapezoid(yields_mass(4)[1], x = yields_mass(4)[0])/integrate.quad(denominator, 0.08, 120)[0]

        he = [ycc_115, ycc_054, ycc_024, ycc_015, ycc_055]

        self.he = he
        self.z = z

        interp = vice.toolkit.interpolation.interp_scheme_1d(xcoords, he)
        self.interp = interp

    def custom_yield(self, zval):

        mh = np.log10(zval/self.solar_z)

        return (self.interp(mh))

###### Type Ia #####
vice.yields.sneia.settings['fe'] = mfeIa
###### Type Ia #####

###### AGB #####
if nitrogen_agb_prefactor == 'oxygen_scaled':
    n_prefactor = mocc / 0.005
else:
    n_prefactor = nitrogen_agb_prefactor

vice.yields.agb.settings["n"] = amplified_agb(element = 'n', study = nitrogen_agb, prefactor = n_prefactor)
vice.yields.agb.settings["c"] = amplified_agb(element = 'c', study = carbon_agb, prefactor = carbon_agb_prefactor)
vice.yields.agb.settings["he"] = helium_agb
###### AGB #####


###### CCSNe #####
if helium_ccsne == 'lc18':

    if net:
        lc18_birth = [2.650e-01, 2.500e-01, 2.400e-01, 2.400e-01]
    else:
        lc18_birth = [0, 0, 0, 0]
    lc18_yields = lc18_interp(birth = lc18_birth, mupper = 120, mlower = 8, net = net, imf = False, rot = ccsne_rot).custom_yield
    vice.yields.ccsne.settings['he'] = lc18_yields

elif helium_ccsne == 'nkt13':

    nkt13_yields = nkt_interp(120, 8, rot = ccsne_rot, net = net, addLC18 = False).custom_yield
    vice.yields.ccsne.settings['he'] = nkt13_yields

elif helium_ccsne == 'nkt13_ext':

    nkt13_yields = nkt_interp(120, 8, rot = ccsne_rot, net = net, addLC18 = True).custom_yield
    vice.yields.ccsne.settings['he'] = nkt13_yields

if nitrogen_ccsne == 'johnson23':
    n_ccsne = 0.024 * mocc
else:
    n_ccsne = nitrogen_ccsne

vice.yields.ccsne.settings['o'] = mocc
vice.yields.ccsne.settings['fe'] = mfecc
vice.yields.ccsne.settings['n'] = n_ccsne
vice.yields.ccsne.settings['c'] = carbon_ccsne
###### CCSNe #####
