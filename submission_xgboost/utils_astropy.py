# import numpy as np
# from astropy import units as u
# from astropy.time import Time
# from astropy.coordinates import GCRS, ITRS, EarthLocation
# from poliastro.bodies import Earth
# from poliastro.twobody import Orbit

# # Define orbital elements
# a = 7000 * u.km       # Semi-major axis
# ecc = 0.1 * u.one     # Eccentricity
# inc = 45 * u.deg      # Inclination
# raan = 60 * u.deg     # RAAN (right ascension of ascending node)
# argp = 30 * u.deg     # Argument of perigee
# nu = 0 * u.deg        # True anomaly

# # Create orbit using poliastro
# orb = Orbit.from_classical(
#     Earth,
#     a,
#     ecc,
#     inc,
#     raan,
#     argp,
#     nu,
#     epoch=Time.now()  # Current time (Astropy Time object)
# )

# # Get position in Geocentric Celestial Reference System (GCRS ≈ ECI)
# position_gcrs = orb.cartesian

# # Transform to International Terrestrial Reference System (ITRS ≈ ECEF)
# position_itrs = position_gcrs.transform_to(ITRS(obstime=Time.now()))

# # Convert to geodetic coordinates (lat/lon/height)
# earth_location = EarthLocation.from_geocentric(
#     position_itrs.x, position_itrs.y, position_itrs.z
# )
# lat = earth_location.lat
# lon = earth_location.lon
# alt = earth_location.height

# print(f"Latitude: {lat:.6f}\nLongitude: {lon:.6f}\nAltitude: {alt:.2f}")










# from poliastro.bodies import Earth
# from poliastro.twobody import Orbit
# from astropy import units as u
# from astropy.time import Time
# from astropy.coordinates import EarthLocation

# # Définir les éléments orbitaux (exemple pour l'ISS)
# i = 1000
# a = initial_states['Semi-major Axis (km)'].loc[i] << u.km          # Demi-grand axe
# ecc = initial_states['Eccentricity'].loc[i]	<< u.one      # Excentricité
# inc = initial_states['Inclination (deg)'].loc[i] << u.deg       # Inclinaison
# raan = initial_states['RAAN (deg)'].loc[i] << u.deg       # Ascension droite du noeud ascendant
# argp = initial_states['Argument of Perigee (deg)'].loc[i]<< u.deg        # Argument du périgée
# nu = initial_states['True Anomaly (deg)'].loc[i] << u.deg           # Anomalie vraie
# epoch = Time(initial_states['Timestamp'].loc[i])       # Date/heure de calcul

# # Créer l'objet Orbite
# orb = Orbit.from_classical(
#     attractor=Earth,
#     a=a,
#     ecc=ecc,
#     inc=inc,
#     raan=raan,
#     argp=argp,
#     nu=nu,
#     epoch=epoch
# )
# sph = orb.represent_as(SphericalRepresentation)
# a = orb.to_ephem()








# def orbital_elements_to_position(a, e, i_deg, omega_deg, raan_deg, nu_deg):
#     i = np.radians(i_deg)
#     omega = np.radians(omega_deg)
#     raan = np.radians(raan_deg)
#     nu = np.radians(nu_deg)

#     r = (a * (1 - e**2)) / (1 + e * np.cos(nu))
#     x_p = r * np.cos(nu)
#     y_p = r * np.sin(nu)
#     r_pqw = np.array([x_p, y_p, 0])

#     def rot_z(theta):
#         c, s = np.cos(theta), np.sin(theta)
#         return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
#     def rot_x(theta):
#         c, s = np.cos(theta), np.sin(theta)
#         return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

#     R = rot_z(raan) @ rot_x(i) @ rot_z(omega)
#     return R @ r_pqw

# def get_julian_date(utc_time):
#     year, month, day = utc_time.year, utc_time.month, utc_time.day
#     a = (14 - month) // 12
#     y = year + 4800 - a
#     m = month + 12*a - 3
#     jdn = day + (153*m + 2)//5 + y*365 + y//4 - y//100 + y//400 - 32045
#     frac = (utc_time.hour + utc_time.minute/60 + utc_time.second/3600) / 24
#     return jdn + frac - 0.5

# def get_gst(utc_time):
#     jd = get_julian_date(utc_time)
#     t = (jd - 2451545.0) / 36525.0
#     gmst_deg = 280.46061837 + 360.98564736628 * (jd - 2451545.0) + 0.000387933 * t**2 - t**3/38710000
#     gmst_deg %= 360
#     return np.radians(gmst_deg)

# def eci_to_ecef(r_eci, gst):
#     c, s = np.cos(-gst), np.sin(-gst)
#     R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
#     return R @ r_eci

# def ecef_to_geodetic(r_ecef):
#     x, y, z = r_ecef
#     a = 6378137.0
#     f = 1/298.257223563
#     b = a * (1 - f)
#     e_sq = (a**2 - b**2) / a**2
#     p = np.hypot(x, y)
#     lon = np.arctan2(y, x)
    
#     lat = np.arctan(z / (p * (1 - e_sq)))
#     for _ in range(10):
#         N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
#         h = p / np.cos(lat) - N
#         denom = 1 - e_sq * N / (N + h)
#         lat_new = np.arctan(z / (p * denom))
#         if abs(lat_new - lat) < 1e-9: break
#         lat = lat_new
    
#     N = a / np.sqrt(1 - e_sq * np.sin(lat))
#     h = p / np.cos(lat) - N
#     return np.degrees(lat), np.degrees(lon), h

# # Example usage
# if __name__ == "__main__":
#     # Orbital elements (example values)
#     a = 7000000  # meters (7000 km)
#     e = 0.1
#     i_deg = 45.0
#     omega_deg = 30.0
#     raan_deg = 60.0
#     nu_deg = 0.0

#     current_utc = datetime.now(timezone.utc)
#     gst = get_gst(current_utc)
#     r_eci = orbital_elements_to_position(a, e, i_deg, omega_deg, raan_deg, nu_deg)
#     r_ecef = eci_to_ecef(r_eci, gst)
#     lat, lon, alt = ecef_to_geodetic(r_ecef)

#     print(f"Latitude: {lat:.6f}°\nLongitude: {lon:.6f}°\nAltitude: {alt:.2f} meters")