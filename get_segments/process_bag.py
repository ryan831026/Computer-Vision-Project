import argparse

from experiment_ros2bag.experiment_bag import InsertionExperimentBag


def __get_argparser():
    """ Configure argument parser """
    parser = argparse.ArgumentParser()

    parser.add_argument( "bagdir", type=str )
    
    parser.add_argument( "--bag-file", type=str, default=None  )
    parser.add_argument( "-t", "--topics", type=str, nargs="+", default=None )
    parser.add_argument( "-r", "--robot", action='store_true' )
    parser.add_argument( "-c", "--camera", action='store_true' )
    parser.add_argument( "-left", "--left", action='store_true' )
    parser.add_argument( "-right", "--right", action='store_true' )
    parser.add_argument( "-isTimeStamp", "--ts", action='store_true' )


    return parser


# __get_argparser

def main( args=None ):
    parser = __get_argparser()

    ARGS = parser.parse_args( args )

    bag = InsertionExperimentBag( ARGS.bagdir )

    side=None
    if ARGS.left:
        side="left"
    elif ARGS.right:
        side="right"

    print("Parsing data in", ARGS.bagdir)
    bag.parse_data(robot=ARGS.robot, camera=ARGS.camera, str_side=side, isGetTimestamps=ARGS.ts)
    
    
# main

if __name__ == "__main__":
    main()

# if __main__