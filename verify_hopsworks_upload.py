from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

api_key = os.getenv('HOPSWORKS_API_KEY')
if not api_key:
    print('HOPSWORKS_API_KEY not set in .env')
    sys.exit(2)

try:
    import hopsworks
    from src.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

    print('Logging into Hopsworks...')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    name = FEATURE_GROUP_NAME
    version = FEATURE_GROUP_VERSION
    print(f'Looking for feature group: {name} v{version}')

    def list_feature_groups():
        """Return all feature groups, handling SDK differences safely."""
        fgs = fs.get_feature_groups()
        return fgs if fgs is not None else []

    def find_latest_feature_group(fg_name):
        """Return the newest feature group with the requested name, if any."""
        matching = [g for g in list_feature_groups() if g.name == fg_name]
        if not matching:
            return None
        return max(matching, key=lambda g: g.version)

    def get_feature_group_or_none(fg_name, fg_version):
        """Some SDK versions return None instead of raising when not found."""
        try:
            return fs.get_feature_group(fg_name, version=fg_version)
        except Exception:
            return None

    def read_feature_group_data(fg):
        """Try the supported read paths and return the first dataframe found."""
        if fg is None:
            return None

        read_attempts = [({}, 'offline'), ({'online': True}, 'online')]
        for read_kwargs, label in read_attempts:
            try:
                df = fg.read(**read_kwargs)
                if df is not None:
                    return df
                print(f'{label.capitalize()} read returned no data.')
            except Exception as read_err:
                print(f'{label.capitalize()} read failed:', read_err)

        return None

    fg = get_feature_group_or_none(name, version)
    if fg is None:
        print(f'Feature group {name} v{version} not found. Searching available versions...')
        try:
            fg = find_latest_feature_group(name)
            if fg is None:
                print('No feature groups found with that name.')
                print('Listing available feature groups:')
                for g in list_feature_groups():
                    print('-', g.name, 'v', g.version)
                sys.exit(4)

            print(f'Using latest available version: {fg.name} v{fg.version}')
        except Exception as e:
            print('Failed while searching fallback feature groups:', e)
            sys.exit(5)

    try:
        df = read_feature_group_data(fg)
        if df is None and fg.version != version:
            print(f'Retrying configured version {version} after fallback read failed...')
            df = read_feature_group_data(get_feature_group_or_none(name, version))

        if df is None:
            print('Could not read any feature group data.')
            print('Listing available feature groups:')
            for g in list_feature_groups():
                print('-', g.name, 'v', g.version)
            sys.exit(6)

        print(f'Found feature group v{fg.version}. Rows: {len(df)}, Columns: {len(df.columns)}')
        print(df.head().to_string(index=False))
    except Exception as e:
        print('Could not fetch feature group directly:', e)
        print('Listing available feature groups:')
        try:
            fgs = list_feature_groups()
            for g in fgs:
                print('-', g.name, 'v', g.version)
        except Exception as e2:
            print('Failed to list feature groups:', e2)

except Exception as e:
    print('Hopsworks client error:', e)
    sys.exit(3)
