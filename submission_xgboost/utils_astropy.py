import math
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants
from org.orekit.utils import IERSConventions
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.orbits import KeplerianOrbit, OrbitType
import math
inertial_frame = FramesFactory.getEME2000()
earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS  # m
# International Terrestrial Reference Frame, earth fixed
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth = OneAxisEllipsoid(
    r_Earth,
    Constants.IERS2010_EARTH_FLATTENING,
    itrf
)
def position_to_geo(pvEME2000, date):
    """
    Converts a position vector (in EME2000 frame) to geodetic coordinates (lat, lon, alt).

    Parameters:
    positionICRF: Vector3D, position vector in ICRF frame.
    date: AbsoluteDate, the date of the position.

    Returns:
    tuple: (latitude, longitude, altitude) in degrees and meters.
    """
    # Transform from EME200 to ICRF
    pvICRF = inertial_frame.getTransformTo(earth_frame,date).transformPVCoordinates(pvEME2000)

    # Transform position from ICRF to ECEF (ITRF)
    transform = earth.getBodyFrame().getTransformTo(itrf, date)
    pvECEF = transform.transformPVCoordinates(pvICRF)
    positionECEF = pvECEF.getPosition()

    # Convert the ECEF position to geodetic coordinates
    geodeticPoint = earth.transform(positionECEF, itrf, date)

    # Extract latitude, longitude, and altitude
    latitude = geodeticPoint.getLatitude()  # radians
    longitude = geodeticPoint.getLongitude()  # radians
    altitude = geodeticPoint.getAltitude()  # meters

    # Convert radians to degrees for latitude and longitude
    latitudeDeg = math.degrees(latitude)
    longitudeDeg = math.degrees(longitude)

    return latitudeDeg, longitudeDeg, altitude

def extract_kepllerian_parameters(sat_state):
    r = sat_state.orbit
    converted_orbit = KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(r))
    latitudeDeg, longitudeDeg, altitude = position_to_geo(sat_state.getPVCoordinates(), sat_state.date)
    res ={
        'A':converted_orbit.getA(),
        'E':converted_orbit.getE(),
    'I':converted_orbit.getI()* 180.0 / math.pi,
    'RAAN':converted_orbit.getRightAscensionOfAscendingNode()* 180.0 / math.pi ,
    'PA':converted_orbit.getPerigeeArgument()* 180.0 / math.pi ,
    'TA':converted_orbit.getTrueAnomaly()* 180.0 / math.pi,
    'Timestamp':r.date,
    'latitudeDeg' : latitudeDeg, 'longitudeDeg':longitudeDeg, 'altitude': altitude}

    
    return res

